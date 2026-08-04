#!/usr/bin/env python3
"""
CLI entry point for the openmix system.
Replaces the monolithic OpenMixer class with a clean pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf

from models import AudioConfig, TrackAnalysis, TransitionLog, CrossfadeDebug
from analyzer import analyze_track
from audio_utils import soft_limit
from crossfader import Crossfader
from mixer import (
    calculate_compatibility,
    smart_track_ordering,
    ensure_smooth_flow,
    align_beats,
    write_transition_log,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_audio_files(folder: Path, formats: tuple, exclude: str = "openmix_output.wav") -> List[Path]:
    files = [f for f in folder.iterdir() if f.suffix.lower() in formats and f.name != exclude]
    files.sort(key=lambda x: x.name.lower())
    logger.info(f"Found {len(files)} audio files")
    return files


def normalize_tracks(tracks: List[TrackAnalysis], target_rms: float = 0.15):
    for t in tracks:
        rms = np.sqrt(np.mean(t.audio_data**2))
        if rms > 0:
            gain = np.clip(target_rms / rms, 0.3, 3.0)
            t.audio_data = t.audio_data * gain


def run(
    input_folder: str,
    output_file: str = "openmix_output.wav",
    sample_rate: int = 44100,
) -> bool:
    folder = Path(input_folder)
    if not folder.is_dir():
        logger.error(f"Input folder does not exist: {input_folder}")
        return False

    config = AudioConfig(sample_rate=sample_rate)
    crossfade_duration = 15.0

    # Discover files
    audio_files = get_audio_files(folder, config.supported_formats)
    if len(audio_files) < 2:
        logger.error("Need at least 2 tracks to create a mix")
        return False

    # Interactive ordering prompt
    print(f"\nFound {len(audio_files)} tracks:")
    for idx, fp in enumerate(audio_files, 1):
        print(f"  {idx}. {fp.name}")

    custom_order = None
    if not sys.stdin.isatty():
        logger.info("Non-interactive mode, using auto-order.")
    else:
        print(f"\nEnter track order (e.g. {' '.join(str(i) for i in range(1, len(audio_files)+1))})")
        user_input = input("or press Enter for auto-order: ").strip()

        if user_input:
            try:
                order = [int(x) for x in user_input.split()]
                if sorted(order) == list(range(1, len(audio_files) + 1)):
                    custom_order = order
                    audio_files = [audio_files[i - 1] for i in order]
                    logger.info(f"Using custom track order: {order}")
                else:
                    print("Invalid order, falling back to auto-order.")
            except ValueError:
                print("Invalid input, falling back to auto-order.")

    # Analyze
    logger.info("Analyzing tracks...")
    analyzed = []
    for fp in audio_files:
        result = analyze_track(fp, config)
        if result:
            analyzed.append(result)

    if len(analyzed) < 2:
        logger.error("Need at least 2 tracks to create a mix")
        return False

    # Order
    if custom_order is None:
        logger.info("Optimizing track order...")
        ordered = smart_track_ordering(analyzed)
    else:
        ordered = analyzed

    # Prepare
    crossfade_samples = int(crossfade_duration * sample_rate)
    transitions_dir = folder / "transitions"
    transitions_dir.mkdir(exist_ok=True)

    normalize_tracks(ordered)
    mixed = ordered[0].audio_data.copy()
    crossfader = Crossfader(sample_rate, crossfade_duration)
    transition_logs: List[TransitionLog] = []

    for i in range(1, len(ordered)):
        current = ordered[i - 1]
        nxt = ordered[i]

        logger.info(f"Mixing: {nxt.file_path.name}")
        compat = calculate_compatibility(current, nxt)
        logger.info(f"  Compatibility score: {compat:.3f}")
        logger.info(f"  Current tempo: {current.tempo:.1f} BPM")
        logger.info(f"  Next tempo: {nxt.tempo:.1f} BPM")

        # Smooth flow — use accumulated mixed as the left side
        _, flow_next, intro_skip = ensure_smooth_flow(current, nxt, crossfade_duration)
        mixed_analysis = _wrap_analysis(current, mixed)
        flow_next_analysis = _wrap_analysis(nxt, flow_next)

        # Beat alignment (skip track1 truncation when mixed contains blended audio from prev transitions)
        aligned_c, aligned_n = align_beats(
            mixed_analysis, flow_next_analysis, crossfade_samples, intro_skip,
            skip_track1=(i > 1),
        )

        # Crossfade — append nxt to accumulated mix
        debug = CrossfadeDebug()
        mixed = crossfader.create(
            aligned_c, aligned_n, crossfade_samples,
            current.tempo, nxt.tempo,
            current.key, nxt.key,
            debug,
        )

        # Save transition clip
        cs = debug.crossfade_section
        if cs is not None:
            _save_transition_clip(cs, transitions_dir, i, sample_rate)

        # Log
        from_key = current.key
        to_key = nxt.key
        key_diff = min(abs(from_key - to_key), 12 - abs(from_key - to_key)) if from_key >= 0 and to_key >= 0 else -1
        mix_position = len(mixed) / sample_rate

        transition_logs.append(TransitionLog(
            transition=i,
            from_track=current.file_path.name,
            to_track=nxt.file_path.name,
            from_tempo=round(current.tempo, 1),
            to_tempo=round(nxt.tempo, 1),
            tempo_diff=round(abs(current.tempo - nxt.tempo), 1),
            tempo_sync_mode=debug.tempo_sync_mode,
            from_key=from_key,
            to_key=to_key,
            key_diff=key_diff,
            key_correction_applied=debug.key_correction,
            compatibility_score=round(compat, 4),
            crossfade_sec=crossfade_duration,
            vocal_segments_outgoing=len(current.vocal_segments),
            vocal_segments_incoming=len(nxt.vocal_segments),
            ducking_applied=debug.ducking_applied,
            ducking_frames=debug.ducking_frames,
            mix_position_sec=round(mix_position, 1),
        ))

    # Final normalization — RMS-based gain for consistent loudness
    logger.info("Applying final volume normalization...")
    final_rms = np.sqrt(np.mean(mixed**2))
    if final_rms > 0:
        mixed = mixed * (0.15 / final_rms)

    mixed = soft_limit(mixed, 0.95)

    # Save (with TPDF dither to mask 16-bit quantization noise)
    output_path = folder / output_file
    rng = np.random.default_rng()
    dither = (rng.uniform(-0.5, 0.5, mixed.shape) + rng.uniform(-0.5, 0.5, mixed.shape)) * (1 / 32768)
    mixed = np.clip(mixed + dither, -1, 1)
    sf.write(str(output_path), mixed, sample_rate)
    write_transition_log(transition_logs, output_path)

    total_dur = len(mixed) / sample_rate
    logger.info(f"Mix created successfully!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Duration: {total_dur:.1f} seconds")
    logger.info(f"Tracks mixed: {len(ordered)}")
    return True


def _wrap_analysis(track: TrackAnalysis, audio: np.ndarray) -> TrackAnalysis:
    """Create a lightweight copy with replaced audio for alignment."""
    return TrackAnalysis(
        file_path=track.file_path,
        duration=len(audio) / track.sample_rate,
        tempo=track.tempo,
        beats=track.beats,
        beat_frames=track.beat_frames,
        key=track.key,
        energy=track.energy,
        energy_variation=track.energy_variation,
        spectral_centroid=track.spectral_centroid,
        spectral_rolloff=track.spectral_rolloff,
        spectral_bandwidth=track.spectral_bandwidth,
        zcr=track.zcr,
        vocal_segments=track.vocal_segments,
        intro_end=track.intro_end,
        outro_start=track.outro_start,
        peak_level=track.peak_level,
        rms_level=track.rms_level,
        audio_data=audio,
        sample_rate=track.sample_rate,
    )


def _save_transition_clip(cs: np.ndarray, transitions_dir: Path, idx: int, sr: int):
    target = int(30 * sr)
    channels = cs.shape[1] if cs.ndim > 1 else 1
    if len(cs) > target:
        center = len(cs) // 2
        half = target // 2
        cs = cs[center - half:center - half + target]
    elif len(cs) < target:
        pad = target - len(cs)
        padding = np.zeros((pad, channels) if cs.ndim > 1 else pad)
        cs = np.concatenate([cs, padding])
    path = transitions_dir / f"transition_{idx}_{idx+1}.wav"
    sf.write(str(path), cs, sr)
    logger.info(f"  Saved transition clip: {path.name} ({len(cs)/sr:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="OpenMix — seamless DJ-style mix")
    parser.add_argument("input_folder", help="Folder containing audio tracks")
    parser.add_argument("-o", "--output", default="openmix_output.wav")
    parser.add_argument("-s", "--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    success = run(args.input_folder, args.output, sample_rate=args.sample_rate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
