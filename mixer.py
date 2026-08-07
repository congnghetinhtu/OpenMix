"""
OpenMix mix assembly: compatibility scoring, track ordering, beat alignment, smooth flow.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from constants import (
    MIN_BEATS_FOR_STRONG,
    SPECTRAL_CENTROID_RANGE,
    STRIDE_STRONG_BEATS,
    TEMPO_RANGE_BPM,
)
from models import TrackAnalysis, TransitionLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

def calculate_compatibility(t1: TrackAnalysis, t2: TrackAnalysis) -> float:
    """Score 0-1 for how well two tracks transition."""
    tempo1, tempo2 = t1.tempo, t2.tempo
    if min(tempo1, tempo2) < 1.0:
        tempo_score = 0.5
    else:
        ratio = max(tempo1, tempo2) / min(tempo1, tempo2)
        if abs(ratio - 2.0) < 0.1 or abs(ratio - 1.5) < 0.1 or abs(ratio - 1.33) < 0.1:
            tempo_score = 0.9
        else:
            tempo_score = max(0, 1 - abs(tempo1 - tempo2) / TEMPO_RANGE_BPM)

    key_distance = min(abs(t1.key - t2.key), 12 - abs(t1.key - t2.key))
    if key_distance == 0:
        key_score = 1.0
    elif key_distance in (5, 7):
        key_score = 0.8
    elif key_distance in (3, 9):
        key_score = 0.7
    else:
        key_score = max(0, 1 - key_distance / 6)

    energy_diff = abs(t1.energy - t2.energy)
    max_energy = max(t1.energy, t2.energy, 0.1)
    energy_score = max(0, 1 - energy_diff / max_energy)

    centroid_diff = abs(t1.spectral_centroid - t2.spectral_centroid)
    spectral_score = max(0, 1 - centroid_diff / SPECTRAL_CENTROID_RANGE)

    return min(1.0, tempo_score * 0.35 + key_score * 0.30 + energy_score * 0.20 + spectral_score * 0.15)


# ---------------------------------------------------------------------------
# Track ordering
# ---------------------------------------------------------------------------

def smart_track_ordering(tracks: List[TrackAnalysis]) -> List[TrackAnalysis]:
    """Greedy reordering for optimal transitions."""
    if len(tracks) <= 1:
        return tracks

    ordered = [tracks[0]]
    remaining = tracks[1:].copy()

    while remaining:
        current = ordered[-1]
        best, best_score = None, -1.0
        for t in remaining:
            score = calculate_compatibility(current, t)
            if score > best_score:
                best_score = score
                best = t
        if best:
            ordered.append(best)
            remaining.remove(best)
            logger.info(f"Next track: {best.file_path.name} (compatibility: {best_score:.2f})")
        else:
            ordered.append(remaining.pop(0))

    return ordered


# ---------------------------------------------------------------------------
# Smooth flow / vocal transitions
# ---------------------------------------------------------------------------

def find_vocal_transition_points(
    vocal1: List[Tuple[float, float]],
    vocal2: List[Tuple[float, float]],
    dur1: float,
    dur2: float,
    beats1: Optional[np.ndarray] = None,
    beats2: Optional[np.ndarray] = None,
    outro_start1: Optional[float] = None,
    intro_end2: Optional[float] = None,
) -> float:
    """Find optimal intro_skip for vocal-to-vocal flow. Returns seconds to skip from track2 start."""

    def snap(
        t: float, beats: Optional[np.ndarray],
        vocal_segments: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        if beats is None or len(beats) == 0:
            return t
        idx = np.argmin(np.abs(beats - t))
        candidate = beats[idx]
        if abs(candidate - t) >= 2.0:
            return t
        # Avoid snapping into a vocal segment
        if vocal_segments and is_in_vocals(candidate, vocal_segments):
            candidates = []
            for offset in [-1, 1, -2, 2]:
                alt_idx = idx + offset
                if 0 <= alt_idx < len(beats):
                    if not is_in_vocals(beats[alt_idx], vocal_segments):
                        candidates.append(beats[alt_idx])
            if candidates:
                valid = [c for c in candidates if abs(c - t) < 2.0]
                if valid:
                    return min(valid, key=lambda x: abs(x - t))
        return candidate

    if not vocal1 or not vocal2:
        return 0.0

    # Use detected boundaries where available, fallback to percentages
    outro_start = outro_start1 if outro_start1 is not None else dur1 * 0.7
    intro_end = intro_end2 if intro_end2 is not None else dur2 * 0.3

    outro_vocals = [(s, e) for s, e in vocal1 if s >= outro_start and e <= dur1]
    intro_vocals = [(s, e) for s, e in vocal2 if s >= 0 and e <= intro_end]

    if not outro_vocals and not intro_vocals:
        return 0.0

    if outro_vocals and not intro_vocals:
        return 0.0

    if not outro_vocals and intro_vocals:
        first = min(intro_vocals, key=lambda x: x[0])
        return snap(max(0, first[0] - 2.0), beats2, intro_vocals)

    # Both have vocals — skip past the first intro vocal segment in track2
    first = min(intro_vocals, key=lambda x: x[0])
    return snap(max(0, first[0] - 1.0), beats2, intro_vocals)


def ensure_smooth_flow(
    track1: TrackAnalysis,
    track2: TrackAnalysis,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (full_audio1, vocal_aligned_audio2, intro_skip_sec) for smooth vocal-to-vocal flow."""
    intro_skip = find_vocal_transition_points(
        track1.vocal_segments,
        track2.vocal_segments,
        track1.duration,
        track2.duration,
        track1.beats,
        track2.beats,
        track1.outro_start,
        track2.intro_end,
    )
    skip_samples = int(intro_skip * track1.sample_rate)
    audio2_aligned = track2.audio_data[skip_samples:] if skip_samples > 0 else track2.audio_data
    logger.info(f"    Vocal-to-vocal transition: skip {intro_skip:.1f}s from track2 start")
    return track1.audio_data, audio2_aligned, intro_skip


# ---------------------------------------------------------------------------
# Beat alignment
# ---------------------------------------------------------------------------

def is_in_vocals(time: float, segments: List[Tuple[float, float]]) -> bool:
    return any(s < time < e for s, e in segments)


def nearest_vocal_boundary(time: float, segments: List[Tuple[float, float]]) -> float:
    boundaries = [b for s, e in segments for b in (s, e)]
    if not boundaries:
        return time
    return boundaries[np.argmin(np.abs(np.array(boundaries) - time))]


def align_beats(
    track1: TrackAnalysis,
    track2: TrackAnalysis,
    crossfade_samples: int,
    audio2_offset: float = 0.0,
    skip_track1: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Snap transition points to strong beats, avoiding vocal cut-through.
    audio2_offset: seconds already trimmed from track2 start (e.g. vocal intro skip).
    skip_track1: if True, skip track1 beat alignment (use when track1 metadata is stale).
    """
    sr = track1.sample_rate
    audio1 = track1.audio_data
    audio2 = track2.audio_data
    beats1 = track1.beats
    beats2 = track2.beats

    crossfade_time = crossfade_samples / sr
    transition_point = (len(audio1) / sr) - crossfade_time

    if len(beats1) > 0 and len(beats2) > 0:
        strong1 = beats1[::STRIDE_STRONG_BEATS] if len(beats1) >= MIN_BEATS_FOR_STRONG else beats1
        strong2 = beats2[::STRIDE_STRONG_BEATS] if len(beats2) >= MIN_BEATS_FOR_STRONG else beats2

        # Determine target phase for track2 alignment
        if not skip_track1:
            beat1_idx = np.argmin(np.abs(strong1 - transition_point))
            beat1_time = strong1[beat1_idx]

            beat_period = 60.0 / track1.tempo if track1.tempo > 0 else 0.5
            if abs(beat1_time - transition_point) < beat_period:
                if is_in_vocals(beat1_time, track1.vocal_segments):
                    beat1_time = nearest_vocal_boundary(beat1_time, track1.vocal_segments)
                    logger.info(f"    Adjusted track1 to vocal boundary: {beat1_time:.2f}s")
                target = int(beat1_time * sr)
                # Ensure enough samples remain for crossfade
                if 0 < target <= len(audio1) and target >= crossfade_samples:
                    audio1 = audio1[:target]
                else:
                    logger.warning("    Track1 beat alignment would leave < crossfade samples, skipping truncation")

        # Phase-align track2: shift so its nearest strong beat matches the transition point
        # strong2 times are in original track2 timeline; audio2 was already trimmed by audio2_offset
        target_phase = transition_point  # where track2 should start
        beat2_candidates_orig = strong2  # in original timeline
        if len(beat2_candidates_orig) > 0:
            # Convert candidates to trimmed-audio timeline
            beat2_trimmed = beat2_candidates_orig - audio2_offset
            # Find best beat to align to target_phase
            best_idx = np.argmin(np.abs(beat2_trimmed - target_phase))
            beat2_shift = beat2_trimmed[best_idx]

            # Offset vocal segments to trimmed-audio timeline for boundary check
            adjusted_vocals = [
                (s - audio2_offset, e - audio2_offset)
                for s, e in track2.vocal_segments
                if e > audio2_offset and s < audio2_offset + len(audio2) / sr
            ]

            if beat2_shift > 0 and is_in_vocals(beat2_shift, adjusted_vocals):
                beat2_shift = nearest_vocal_boundary(beat2_shift, adjusted_vocals)
                logger.info(f"    Adjusted track2 to vocal boundary: {beat2_shift:.2f}s")
            beat2_samples = int(beat2_shift * sr)
            if 0 < beat2_samples < len(audio2):
                audio2 = audio2[beat2_samples:]

    return audio1, audio2


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def write_transition_log(logs: List[TransitionLog], output_path: Path):
    if not logs:
        return
    log_path = output_path.with_suffix('.csv')
    fieldnames = [
        'transition', 'from_track', 'to_track', 'from_tempo', 'to_tempo',
        'tempo_diff', 'tempo_sync_mode', 'from_key', 'to_key', 'key_diff',
        'compatibility_score', 'crossfade_sec',
        'vocal_segments_outgoing', 'vocal_segments_incoming',
        'ducking_applied', 'ducking_frames', 'mix_position_sec',
        'phase_correlation', 'zero_crossing_aligned', 'phase_inverted',
    ]
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            writer.writerow(log.to_dict())
    logger.info(f"Transition log saved: {log_path}")
