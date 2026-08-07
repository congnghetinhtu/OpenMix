"""
OpenMix crossfade creation: tempo sync, vocal-aware blending.
Stateful per-transition — creates a fresh instance per transition pair.
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from audio_utils import (
    align_to_zero_crossings,
    apply_fade,
    compute_phase_correlation,
    make_equal_power_crossfade,
    time_stretch,
)
from constants import (
    BLEND_SAMPLES,
    FILTER_MIN_HZ,
    FILTER_ORDER,
    LOWPASS_START_HZ,
    PHASE_CORRELATION_WINDOW,
    PHASE_INVERT_THRESHOLD,
    SOFT_LIMIT_CEILING,
    TEMPO_MAX_STRETCH_INVISIBLE,
    TEMPO_MAX_STRETCH_LARGE,
    TEMPO_MAX_STRETCH_SMALL,
)
from models import CrossfadeDebug

logger = logging.getLogger(__name__)


class Crossfader:
    def __init__(self, sample_rate: int, crossfade_duration: float):
        self.sr = sample_rate
        self.crossfade_duration = crossfade_duration
        self.max_tempo_samples = int(sample_rate * crossfade_duration * 0.9)

    def create(
        self,
        track1_audio: np.ndarray,
        track2_audio: np.ndarray,
        crossfade_samples: int,
        track1_tempo: Optional[float] = None,
        track2_tempo: Optional[float] = None,
        debug_log: Optional[CrossfadeDebug] = None,
    ) -> np.ndarray:
        if debug_log is None:
            debug_log = CrossfadeDebug()

        requested_cf = crossfade_samples
        crossfade_samples = min(crossfade_samples, len(track1_audio), len(track2_audio))
        if crossfade_samples < requested_cf:
            logger.warning(f"Crossfade shortened: {requested_cf} -> {crossfade_samples} samples (track too short)")

        # Original transition window length — tempo sync below may change segment lengths,
        # but the prefix cutoff must stay anchored to the original track1 timeline.
        window_samples = crossfade_samples

        track1_end = track1_audio[-crossfade_samples:].copy()
        track2_start = track2_audio[:crossfade_samples].copy()

        # 5s lead-in before the crossfade so the low-pass ramps in early
        lead_samples = int(self.sr * 5.0)
        lead = track1_audio[-crossfade_samples - lead_samples:-crossfade_samples].copy()

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

            track1_end = self._apply_gradual_tempo_ramp(
                track1_end, track1_tempo, track2_tempo, is_outro=True
            )
            track2_start = self._apply_gradual_tempo_ramp(
                track2_start, track2_tempo, track1_tempo, is_outro=False
            )
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
            window_samples = actual_cf

        # Low-pass filter the OUTGOING track to mask the transition, starting 5s before
        filter_input = np.concatenate([lead.astype(track1_end.dtype), track1_end]) if len(lead) else track1_end
        filtered = self._apply_transition_filter(filter_input, crossfade_samples, center_offset=len(lead))
        if len(lead):
            lead_filtered = filtered[:len(lead)]
            track1_end = filtered[len(lead):]
        else:
            lead_filtered = None
            track1_end = filtered

        # Phase alignment: zero-crossing boundary + correlation check
        t1_analysis = track1_end if track1_end.ndim == 1 else np.mean(track1_end, axis=1)
        t2_analysis = track2_start if track2_start.ndim == 1 else np.mean(track2_start, axis=1)

        # Compute phase correlation at boundary
        corr = compute_phase_correlation(
            t1_analysis[-PHASE_CORRELATION_WINDOW:],
            t2_analysis[:PHASE_CORRELATION_WINDOW],
        )
        debug_log.phase_correlation = corr

        # Align both tracks to zero crossings at the boundary
        track1_end, track2_start, zc_shift = align_to_zero_crossings(track1_end, track2_start)
        if zc_shift > 0:
            debug_log.zero_crossing_aligned = True
            logger.info(f"    Zero-crossing aligned: removed {zc_shift} samples at boundary")

        # Invert if strongly out of phase
        if corr < PHASE_INVERT_THRESHOLD:
            track2_start = -track2_start
            debug_log.phase_inverted = True
            logger.info(f"    Phase inverted track2 (correlation: {corr:.3f})")
        elif abs(corr) > 0.3:
            logger.info(f"    Phase correlation at boundary: {corr:.3f}")

        # Update crossfade length after zero-crossing alignment
        actual_cf = min(len(track1_end), len(track2_start))
        if actual_cf < crossfade_samples:
            track1_end = track1_end[:actual_cf]
            track2_start = track2_start[:actual_cf]
            crossfade_samples = actual_cf
            window_samples = actual_cf

        # Crossfade curves (match actual segment length)
        fade_out, fade_in = make_equal_power_crossfade(crossfade_samples)

        # Vocal-aware blending
        crossfade_section = self._vocal_aware_crossfade(
            track1_end, track2_start, fade_out, fade_in, debug_log
        )

        debug_log.crossfade_section = crossfade_section

        # Blend boundary: overlap unmodified tail into modified crossfade start (prevents click)
        blend = BLEND_SAMPLES
        if len(track1_audio) > window_samples + blend:
            if lead_filtered is not None and len(lead_filtered) >= blend:
                tail = lead_filtered[-blend:].copy()
            else:
                tail = track1_audio[-(crossfade_samples + blend):-crossfade_samples].copy()
            head = crossfade_section[:blend].copy()
            fade = np.linspace(0, 1, blend)
            if tail.ndim > 1:
                fade = fade[:, None]
            crossfade_section[:blend] = tail * (1 - fade) + head * fade

        if lead_filtered is not None:
            result = np.concatenate([
                track1_audio[:-(window_samples + len(lead_filtered))],
                lead_filtered,
                crossfade_section,
                track2_audio[crossfade_samples:],
            ])
        else:
            result = np.concatenate([
                track1_audio[:-crossfade_samples],
                crossfade_section,
                track2_audio[crossfade_samples:],
            ])
        return result

    # -- internal helpers --

    def _apply_gradual_tempo_ramp(
        self, audio: np.ndarray, original_tempo: float, target_tempo: float, is_outro: bool
    ) -> np.ndarray:
        """Apply gradual tempo ramp across segment for smooth beat alignment.

        Splits audio into overlapping chunks and applies progressively different
        stretch factors. Outgoing track ramps toward incoming tempo; incoming
        track ramps from outgoing tempo back to natural. Beats meet at center.
        """
        tempo_diff = abs(original_tempo - target_tempo)
        if tempo_diff < 0.5:
            return audio

        if len(audio) > self.max_tempo_samples:
            logger.info("    Skipping heavy tempo sync for long section")
            return audio

        # Target stretch factor: full correction to match tempos
        if is_outro:
            target_factor = original_tempo / target_tempo
        else:
            target_factor = target_tempo / original_tempo

        target_factor = np.clip(target_factor, 0.75, 1.35)

        if abs(target_factor - 1.0) < 0.01:
            return audio

        # Short segment: single uniform stretch
        if len(audio) < 4096:
            try:
                result = time_stretch(audio, rate=target_factor)
                if len(result) != len(audio):
                    result = signal.resample(result, len(audio))
                logger.info(f"    Applied uniform tempo sync: {target_factor:.4f}x (short segment)")
                return result
            except Exception as e:
                logger.warning(f"    Uniform tempo sync failed: {e}")
                return audio

        # Gradual ramp via overlap-add chunking
        n_chunks = 16
        chunk_size = max(1024, len(audio) // n_chunks)
        overlap = chunk_size // 2

        if is_outro:
            factors = np.linspace(1.0, target_factor, n_chunks)
        else:
            factors = np.linspace(target_factor, 1.0, n_chunks)

        result = np.zeros_like(audio)
        win_sum = np.zeros(len(audio))

        for i in range(n_chunks):
            start = i * (chunk_size - overlap)
            end = min(start + chunk_size, len(audio))
            if start >= len(audio):
                break

            chunk = audio[start:end]
            size = end - start
            factor = factors[i]

            if abs(factor - 1.0) < 0.005:
                w = np.hanning(size * 2)[:size]
                if chunk.ndim > 1:
                    w = w[:, None]
                result[start:end] += chunk * w
                win_sum[start:end] += w.squeeze()
                continue

            try:
                stretched = time_stretch(chunk, rate=factor)
                if len(stretched) != size:
                    stretched = signal.resample(stretched, size)
                w = np.hanning(size * 2)[:size]
                if stretched.ndim > 1:
                    w = w[:, None]
                result[start:end] += stretched * w
                win_sum[start:end] += w.squeeze()
            except Exception as e:
                logger.warning(f"    Tempo ramp chunk {i} failed: {e}")
                w = np.hanning(size * 2)[:size]
                if chunk.ndim > 1:
                    w = w[:, None]
                result[start:end] += chunk * w
                win_sum[start:end] += w.squeeze()

        win_sum = np.maximum(win_sum, 1e-10)
        if result.ndim > 1:
            win_sum = win_sum[:, None]
        result /= win_sum

        logger.info(f"    Applied gradual tempo ramp: {original_tempo:.1f} -> {target_tempo:.1f} BPM (factor: {target_factor:.3f})")
        return result

    def _apply_transition_filter(
        self, audio: np.ndarray, crossfade_samples: int, center_offset: int = 0
    ) -> np.ndarray:
        """Smooth low-pass filter during crossfade center.
        RMS-normalized per segment to prevent volume loss.
        With center_offset (lead-in samples), the low-pass ramps in
        starting center_offset samples before the transition point.
        """
        n = len(audio)
        nyquist = self.sr / 2
        min_freq = LOWPASS_START_HZ
        if center_offset > 0:
            center = center_offset
            active_half = center_offset
            one_sided = True
        else:
            center = n // 2
            active_half = n // 3
            one_sided = False

        result = audio.astype(np.float64).copy()
        seg_len = 8192
        hop = seg_len // 2
        order = FILTER_ORDER

        right_bound = n if one_sided else center + active_half

        for start in range(max(0, center - active_half - seg_len), min(n - seg_len, right_bound), hop):
            end = start + seg_len
            mid = start + seg_len // 2

            if one_sided:
                dist = np.clip((center - mid) / active_half, 0.0, 1.0)
            else:
                dist = abs(mid - center) / active_half
                dist = min(dist, 1.0)
            cutoff = min_freq + (nyquist - min_freq) * dist
            cutoff = np.clip(cutoff, FILTER_MIN_HZ, nyquist - 100.0)

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
        track2_orig = track2_start.copy()
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
                def _apply_duck(audio, curve):
                    if audio.ndim > 1:
                        return audio * curve[:, None]
                    return audio * curve
                track1_ducked = _apply_duck(track1_end, duck_curve)
                track2_start = _apply_duck(track2_start, duck_curve)

                debug_log.ducking_applied = True
                debug_log.ducking_frames = int(np.sum(duck_mask))
                logger.info(f"    Vocal overlap prevention: ducked {np.sum(duck_mask)} frames")
            else:
                debug_log.ducking_applied = False
                debug_log.ducking_frames = 0
                track1_ducked = track1_end

        except Exception as e:
            debug_log.ducking_applied = False
            debug_log.ducking_frames = 0
            logger.debug(f"Vocal overlap detection failed: {e}, using standard crossfade")
            track1_ducked = track1_end
            track2_start = track2_orig

        # Phase safety net: if still negatively correlated after ducking, scale down
        # to reduce cancellation damage (zero-crossing alignment handles most cases)
        post_duck_corr = compute_phase_correlation(
            track1_ducked if track1_ducked.ndim == 1 else np.mean(track1_ducked, axis=1),
            track2_start if track2_start.ndim == 1 else np.mean(track2_start, axis=1),
        )
        if post_duck_corr < -0.3:
            attenuation = np.sqrt(1.0 - abs(post_duck_corr))
            track1_ducked = track1_ducked * attenuation
            track2_start = track2_start * attenuation
            logger.info(f"    Phase safety: attenuated both tracks by {attenuation:.3f} (corr: {post_duck_corr:.3f})")

        crossfaded = apply_fade(track1_ducked, fade_out) + apply_fade(track2_start, fade_in)

        peak = np.max(np.abs(crossfaded))
        if peak > SOFT_LIMIT_CEILING:
            crossfaded = crossfaded * (SOFT_LIMIT_CEILING / peak)

        return crossfaded
