"""
OpenMix crossfade creation: tempo sync, key correction, vocal-aware blending.
Stateful per-transition — creates a fresh instance per transition pair.
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from models import CrossfadeDebug
from audio_utils import (
    apply_fade,
    time_stretch,
    make_equal_power_crossfade,
)

logger = logging.getLogger(__name__)


class Crossfader:
    def __init__(self, sample_rate: int, crossfade_duration: float):
        self.sr = sample_rate
        self.crossfade_duration = crossfade_duration
        self.max_tempo_samples = int(sample_rate * crossfade_duration * 0.9)
        self.max_key_samples = int(sample_rate * crossfade_duration * 0.9)

    def create(
        self,
        track1_audio: np.ndarray,
        track2_audio: np.ndarray,
        crossfade_samples: int,
        track1_tempo: Optional[float] = None,
        track2_tempo: Optional[float] = None,
        track1_key: Optional[int] = None,
        track2_key: Optional[int] = None,
        debug_log: Optional[CrossfadeDebug] = None,
    ) -> np.ndarray:
        if debug_log is None:
            debug_log = CrossfadeDebug()

        requested_cf = crossfade_samples
        crossfade_samples = min(crossfade_samples, len(track1_audio), len(track2_audio))
        if crossfade_samples < requested_cf:
            logger.warning(f"Crossfade shortened: {requested_cf} -> {crossfade_samples} samples (track too short)")

        track1_end = track1_audio[-crossfade_samples:].copy()
        track2_start = track2_audio[:crossfade_samples].copy()

        # Tempo sync (may change segment lengths)
        tempo_diff = abs(track1_tempo - track2_tempo) if track1_tempo and track2_tempo else 0
        if track1_tempo and track2_tempo and tempo_diff > 1.5:
            if tempo_diff < 4:
                mode = "invisible"
            elif track2_tempo > track1_tempo:
                mode = "increasing"
            else:
                mode = "decreasing"
            debug_log.tempo_sync_mode = mode
            logger.info(f"  Creating {mode} tempo transition: {track1_tempo:.1f} -> {track2_tempo:.1f} BPM")

            if tempo_diff < 10:
                track1_end = self._apply_invisible_tempo_sync(track1_end, track1_tempo, track2_tempo, is_outro=True)
                track2_start = self._apply_invisible_tempo_sync(track2_start, track2_tempo, track1_tempo, is_outro=False)
            else:
                mid = (track1_tempo + track2_tempo) / 2
                track1_end = self._apply_invisible_tempo_sync(track1_end, track1_tempo, mid, is_outro=True)
                track2_start = self._apply_invisible_tempo_sync(track2_start, track2_tempo, mid, is_outro=False)
        else:
            debug_log.tempo_sync_mode = 'none'
            logger.info(f"  Tempo difference minimal ({tempo_diff:.1f} BPM), preserving natural flow")

        # After tempo sync, segments may differ in length — trim to minimum
        actual_cf = min(len(track1_end), len(track2_start))
        if actual_cf < crossfade_samples:
            logger.info(f"    Tempo sync shortened segments: {crossfade_samples} -> {actual_cf} samples")
            track1_end = track1_end[:actual_cf]
            track2_start = track2_start[:actual_cf]
            crossfade_samples = actual_cf

        # Key correction — crossfade window only (full track stays unshifted)
        if track1_key is not None and track2_key is not None:
            key_diff = (track2_key - track1_key) % 12
            if key_diff > 6:
                key_diff -= 12
            debug_log.key_correction = abs(key_diff) > 1 and abs(key_diff) not in (5, 7)
            track1_end = self._apply_key_correction(track1_end, track1_key, track2_key, is_outro=True)
            track2_start = self._apply_key_correction(track2_start, track2_key, track1_key, is_outro=False)

        # Low-pass filter the OUTGOING track to mask the transition
        track1_end = self._apply_transition_filter(track1_end, crossfade_samples)

        # Crossfade curves (match actual segment length)
        fade_out, fade_in = make_equal_power_crossfade(crossfade_samples)

        # Vocal-aware blending
        crossfade_section = self._vocal_aware_crossfade(
            track1_end, track2_start, fade_out, fade_in, debug_log
        )

        debug_log.crossfade_section = crossfade_section

        # Blend boundary: overlap unmodified tail into modified crossfade start (prevents click)
        blend = 256
        if len(track1_audio) > crossfade_samples + blend:
            tail = track1_audio[-(crossfade_samples + blend):-crossfade_samples].copy()
            head = crossfade_section[:blend].copy()
            fade = np.linspace(0, 1, blend)
            if tail.ndim > 1:
                fade = fade[:, None]
            crossfade_section[:blend] = tail * (1 - fade) + head * fade

        result = np.concatenate([
            track1_audio[:-crossfade_samples],
            crossfade_section,
            track2_audio[crossfade_samples:],
        ])
        return result

    # -- internal helpers --

    def _apply_invisible_tempo_sync(
        self, audio: np.ndarray, original_tempo: float, target_tempo: float, is_outro: bool
    ) -> np.ndarray:
        tempo_diff = abs(original_tempo - target_tempo)
        if tempo_diff < 0.5:
            return audio

        if len(audio) > self.max_tempo_samples:
            logger.info("    Skipping heavy tempo sync for long section")
            return audio

        max_change = min(0.02, tempo_diff / original_tempo * 0.15)
        if is_outro:
            factor = 1.0 + max_change * 0.5 if target_tempo > original_tempo else 1.0 - max_change * 0.5
        else:
            factor = 1.0 - max_change * 0.3 if original_tempo > target_tempo else 1.0 + max_change * 0.3

        if abs(factor - 1.0) <= 0.005:
            return audio

        try:
            result = time_stretch(audio, rate=factor)
            logger.info(f"    Applied invisible tempo sync: {factor:.4f}x stretch")
            return result

        except Exception as e:
            logger.warning(f"    Invisible tempo sync failed: {e}, using original")
            return audio

    def _apply_key_correction(
        self, audio: np.ndarray, source_key: int, target_key: int, is_outro: bool
    ) -> np.ndarray:
        key_diff = (target_key - source_key) % 12
        if key_diff > 6:
            key_diff -= 12

        if abs(key_diff) <= 1 or abs(key_diff) in (5, 7):
            return audio

        if len(audio) > self.max_key_samples:
            return audio

        shift = key_diff * 0.25
        shift = np.clip(shift, -2, 2)

        if abs(shift) <= 0.1:
            return audio

        try:
            shifted = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=shift)
            logger.info(f"    Applied key correction: shift {shift:.2f} semitones")
            return shifted
        except Exception as e:
            logger.warning(f"    Key correction failed: {e}, using original")
            return audio

    def _apply_transition_filter(
        self, audio: np.ndarray, crossfade_samples: int
    ) -> np.ndarray:
        """Smooth low-pass filter during crossfade center.
        RMS-normalized per segment to prevent volume loss.
        """
        n = len(audio)
        nyquist = self.sr / 2
        min_freq = 3500.0
        center = n // 2
        active_half = n // 3

        result = audio.astype(np.float64).copy()
        seg_len = 8192
        hop = seg_len // 2
        order = 2

        for start in range(max(0, center - active_half - seg_len), min(n - seg_len, center + active_half), hop):
            end = start + seg_len
            mid = start + seg_len // 2

            dist = abs(mid - center) / active_half
            dist = min(dist, 1.0)
            cutoff = min_freq + (nyquist - min_freq) * dist
            cutoff = np.clip(cutoff, 200.0, nyquist - 100.0)

            wn = cutoff / nyquist
            sos = signal.butter(order, wn, btype='low', output='sos')

            if audio.ndim == 1:
                segment = audio[start:end].astype(np.float64)
                rms_orig = np.sqrt(np.mean(segment**2)) + 1e-10
                filtered = signal.sosfiltfilt(sos, segment)
                rms_filt = np.sqrt(np.mean(filtered**2)) + 1e-10
                filtered = filtered * (rms_orig / rms_filt)
                fade = np.hanning(end - start)
                result[start:end] = result[start:end] * (1 - fade) + filtered * fade
            else:
                for ch in range(audio.shape[1]):
                    segment = audio[start:end, ch].astype(np.float64)
                    rms_orig = np.sqrt(np.mean(segment**2)) + 1e-10
                    filtered = signal.sosfiltfilt(sos, segment)
                    rms_filt = np.sqrt(np.mean(filtered**2)) + 1e-10
                    filtered = filtered * (rms_orig / rms_filt)
                    fade = np.hanning(end - start)
                    result[start:end, ch] = result[start:end, ch] * (1 - fade) + filtered * fade

        return result.astype(audio.dtype)

    def _vocal_aware_crossfade(
        self,
        track1_end: np.ndarray,
        track2_start: np.ndarray,
        fade_out: np.ndarray,
        fade_in: np.ndarray,
        debug_log: CrossfadeDebug,
    ) -> np.ndarray:
        try:
            t1_mono = track1_end if track1_end.ndim == 1 else np.mean(track1_end, axis=1)
            t2_mono = track2_start if track2_start.ndim == 1 else np.mean(track2_start, axis=1)

            # Use direct RMS energy instead of HPSS (HPSS on short segments produces artifacts)
            frame_length = 2048
            hop_length = 512

            v1_env = librosa.feature.rms(y=t1_mono, frame_length=frame_length, hop_length=hop_length)[0]
            v2_env = librosa.feature.rms(y=t2_mono, frame_length=frame_length, hop_length=hop_length)[0]

            v1_up = np.interp(np.linspace(0, 1, len(track1_end)), np.linspace(0, 1, len(v1_env)), v1_env)
            v2_up = np.interp(np.linspace(0, 1, len(track2_start)), np.linspace(0, 1, len(v2_env)), v2_env)

            v1_up /= np.max(v1_up) + 1e-8
            v2_up /= np.max(v2_up) + 1e-8

            overlap = v1_up * v2_up
            duck_curve = np.ones(len(track1_end))
            duck_threshold = 0.1
            duck_mask = overlap > duck_threshold

            if np.any(duck_mask):
                duck_strength = 0.6
                duck_curve[duck_mask] = 1.0 - (1.0 - duck_strength) * overlap[duck_mask] / np.max(overlap[duck_mask])
                if len(duck_curve) > 100:
                    duck_curve = gaussian_filter1d(duck_curve, sigma=50)
                    duck_curve = np.clip(duck_curve, 0.5, 1.0)

            # Apply symmetric ducking to both tracks (prevents muddy buildup)
            apply_duck = lambda audio, curve: audio * curve[:, None] if audio.ndim > 1 else audio * curve
            track1_ducked = apply_duck(track1_end, duck_curve)
            track2_start = apply_duck(track2_start, duck_curve)

            debug_log.ducking_applied = True
            debug_log.ducking_frames = int(np.sum(duck_mask))
            logger.info(f"    Vocal overlap prevention: ducked {np.sum(duck_mask)} frames")

        except Exception as e:
            debug_log.ducking_applied = False
            debug_log.ducking_frames = 0
            logger.debug(f"Vocal overlap detection failed: {e}, using standard crossfade")
            track1_ducked = track1_end

        crossfaded = apply_fade(track1_ducked, fade_out) + apply_fade(track2_start, fade_in)

        peak = np.max(np.abs(crossfaded))
        if peak > 0.95:
            crossfaded = crossfaded * (0.95 / peak)

        return crossfaded
