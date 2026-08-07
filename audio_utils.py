"""
OpenMix low-level audio processing utilities.
Pure functions — no class state. Maps to Swift enum AudioHelpers.
"""

import librosa
import numpy as np
from scipy import signal

from constants import (
    NORMALIZE_GAIN_MAX,
    NORMALIZE_GAIN_MIN,
    SOFT_LIMIT_CEILING,
    ZERO_CROSSING_SEARCH,
)


def normalize_audio(audio: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
    """Normalize audio to stable volume with dynamics preservation."""
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    if rms <= 0 or peak <= 0:
        return audio

    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / rms
    gain = min(gain, 0.9 / peak)
    gain = np.clip(gain, NORMALIZE_GAIN_MIN, NORMALIZE_GAIN_MAX)

    normalized = audio * gain

    new_peak = np.max(np.abs(normalized))
    if new_peak > 0.85:
        threshold = 0.85
        ratio = 0.3
        mask = np.abs(normalized) > threshold
        excess = np.abs(normalized) - threshold
        compressed_excess = excess * ratio
        normalized = np.where(
            mask,
            np.sign(normalized) * (threshold + compressed_excess),
            normalized,
        )

    return normalized


def analysis_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono for analysis."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def apply_fade(audio: np.ndarray, fade: np.ndarray) -> np.ndarray:
    """Apply a time-based fade curve to mono or multi-channel audio."""
    if audio.ndim == 1:
        return audio * fade
    return audio * fade[:, None]


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch audio while preserving channel layout.
    For multichannel, stretches first channel to determine target length,
    then resamples all channels to that length (avoids L/R phase drift).
    """
    if len(audio) < 32 or abs(rate - 1.0) < 1e-6:
        return audio

    if audio.ndim == 1:
        return librosa.effects.time_stretch(audio, rate=rate)

    # Multichannel: stretch channel 0 to determine target length,
    # then resample all channels to that length (avoids L/R phase drift).
    ref = librosa.effects.time_stretch(audio[:, 0], rate=rate)
    new_length = len(ref)
    channels = [ref]
    for ch in range(1, audio.shape[1]):
        stretched = librosa.effects.time_stretch(audio[:, ch], rate=rate)
        if len(stretched) != new_length:
            stretched = signal.resample(stretched, new_length)
        channels.append(stretched)
    return np.stack(channels, axis=1)


def make_equal_power_crossfade(n: int):
    """Create equal-power crossfade curves with quintic smoothstep."""
    fade_curve = np.linspace(0, 1, n)
    smooth = fade_curve ** 3 * (fade_curve * (fade_curve * 6 - 15) + 10)
    fade_out = np.cos(smooth * np.pi / 2)
    fade_in = np.sin(smooth * np.pi / 2)
    fade_out[0] = 1.0
    fade_out[-1] = 0.0
    fade_in[0] = 0.0
    fade_in[-1] = 1.0
    return fade_out, fade_in


def soft_limit(audio: np.ndarray, ceiling: float = SOFT_LIMIT_CEILING) -> np.ndarray:
    """Apply gentle peak limiting."""
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        return audio * (ceiling / peak)
    return audio


def compute_phase_correlation(t1_tail: np.ndarray, t2_head: np.ndarray) -> float:
    """Compute Pearson correlation between tail of track1 and head of track2.

    Both inputs are mixed down to mono if multichannel.
    Returns float in [-1.0, 1.0]. 1.0 = perfectly correlated, -1.0 = inverted.
    """
    if t1_tail.ndim > 1:
        t1_tail = np.mean(t1_tail, axis=1)
    if t2_head.ndim > 1:
        t2_head = np.mean(t2_head, axis=1)

    n = min(len(t1_tail), len(t2_head))
    if n < 64:
        return 0.0

    a = t1_tail[-n:].astype(np.float64)
    b = t2_head[:n].astype(np.float64)

    a_std = np.std(a)
    b_std = np.std(b)
    if a_std < 1e-10 or b_std < 1e-10:
        return 0.0

    a = (a - np.mean(a)) / a_std
    b = (b - np.mean(b)) / b_std

    return float(np.mean(a * b))


def find_zero_crossing(audio: np.ndarray, center: int, search_range: int = ZERO_CROSSING_SEARCH) -> int:
    """Find nearest zero crossing to `center` within ±search_range samples.

    Returns sample index of the zero crossing closest to center.
    If no zero crossing found, returns center unchanged.
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    n = len(audio)
    start = max(0, center - search_range)
    end = min(n - 1, center + search_range)

    best_idx = center
    best_dist = search_range + 1

    for i in range(start, end):
        if audio[i] * audio[i + 1] <= 0:
            # Zero crossing found — pick the sample with smaller absolute value
            if abs(audio[i]) < abs(audio[i + 1]):
                idx = i
            else:
                idx = i + 1
            dist = abs(idx - center)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

    return best_idx


def align_to_zero_crossings(
    t1_end: np.ndarray, t2_start: np.ndarray
) -> tuple:
    """Align crossfade boundary to zero crossings on both tracks.

    Finds the nearest zero crossing near the end of t1_end and the start
    of t2_start, then trims both to land exactly on those crossings.
    Returns (aligned_t1_end, aligned_t2_start, samples_removed).
    """
    if t1_end.ndim > 1:
        t1_mono = np.mean(t1_end, axis=1)
        t2_mono = np.mean(t2_start, axis=1)
    else:
        t1_mono = t1_end
        t2_mono = t2_start

    # Find zero crossing near end of track1
    t1_zc = find_zero_crossing(t1_mono, len(t1_mono) - 1)
    # Find zero crossing near start of track2
    t2_zc = find_zero_crossing(t2_mono, 0)

    # Trim: keep t1 up to and including its zero crossing
    # Trim: start t2 from its zero crossing
    new_t1 = t1_end[:t1_zc + 1] if t1_zc < len(t1_end) - 1 else t1_end
    new_t2 = t2_start[t2_zc:] if t2_zc > 0 else t2_start

    samples_removed = (len(t1_end) - len(new_t1)) + (len(t2_start) - len(new_t2))
    return new_t1, new_t2, samples_removed
