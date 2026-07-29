"""
OpenMix low-level audio processing utilities.
Pure functions — no class state. Maps to Swift enum AudioHelpers.
"""

import numpy as np
import librosa
from scipy import signal


def normalize_audio(audio: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
    """Normalize audio to stable volume with dynamics preservation."""
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    if rms <= 0 or peak <= 0:
        return audio

    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / rms
    gain = min(gain, 0.9 / peak)
    gain = np.clip(gain, 0.3, 2.0)

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
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        if len(stretched) != len(audio):
            stretched = signal.resample(stretched, len(audio))
        return stretched

    # Multichannel: stretch channel 0 to determine output length
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


def soft_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Apply gentle peak limiting."""
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        return audio * (ceiling / peak)
    return audio
