"""
OpenMix audio track analysis: tempo, key, vocals, energy, spectral features.
Pure analysis logic — no mixing/crossfade state.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import librosa
import numpy as np
from scipy import signal

from models import AudioConfig, TrackAnalysis
from audio_utils import normalize_audio, analysis_mono

logger = logging.getLogger(__name__)


def analyze_track(file_path: Path, config: AudioConfig) -> Optional[TrackAnalysis]:
    """Analyze a single audio file and return full feature set."""
    try:
        logger.info(f"Analyzing: {file_path.name}")

        y, sr = librosa.load(str(file_path), sr=config.sample_rate, mono=False)
        if y.ndim > 1:
            y = y.T

        y_normalized = normalize_audio(y)
        mono = analysis_mono(y_normalized)

        max_samples = int(sr * config.max_analysis_seconds)
        focus = mono[:max_samples] if len(mono) > max_samples else mono

        if len(focus) < 2048:
            logger.warning(f"{file_path.name}: very short signal, using fallback")
            tempo = 120.0
            beats = np.array([])
            beat_frames = np.array([])
        else:
            tempo, beats = librosa.beat.beat_track(y=focus, sr=sr, units='time')
            tempo = float(np.asarray(tempo).reshape(-1)[0])
            beat_frames = librosa.beat.beat_track(y=focus, sr=sr, units='frames')[1]

        n_fft = max(2, min(2048, len(focus)))
        hop_length = max(1, min(512, max(1, n_fft // 4)))

        chroma = librosa.feature.chroma_stft(y=focus, sr=sr, n_fft=n_fft, hop_length=hop_length)
        key_profile = np.mean(chroma, axis=1)
        key = int(np.argmax(key_profile))

        rms = librosa.feature.rms(y=focus, frame_length=n_fft, hop_length=hop_length)[0]
        energy = float(np.mean(rms))
        energy_variation = float(np.std(rms))

        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=focus, sr=sr, n_fft=n_fft, hop_length=hop_length)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=focus, sr=sr, n_fft=n_fft, hop_length=hop_length)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=focus, sr=sr, n_fft=n_fft, hop_length=hop_length)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(focus)))

        vocal_segments = detect_vocals_edges(mono, sr, config)
        intro_end, outro_start = detect_intro_outro(focus, sr, beats)

        peak_level = float(np.max(np.abs(y_normalized)))
        rms_level = float(np.sqrt(np.mean(y_normalized**2)))

        return TrackAnalysis(
            file_path=file_path,
            duration=len(y) / sr,
            tempo=tempo,
            beats=beats,
            beat_frames=beat_frames,
            key=key,
            energy=energy,
            energy_variation=energy_variation,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_bandwidth=spectral_bandwidth,
            zcr=zcr,
            vocal_segments=vocal_segments,
            intro_end=intro_end,
            outro_start=outro_start,
            peak_level=peak_level,
            rms_level=rms_level,
            audio_data=y,
            sample_rate=sr,
        )

    except Exception as e:
        logger.error(f"Error analyzing {file_path.name}: {e}")
        return None


def detect_vocals(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocal segments using harmonic-percussive separation."""
    try:
        if len(y) < 2048:
            return []

        n_fft = max(2, min(2048, len(y)))
        hop_length = max(1, min(512, max(1, n_fft // 4)))

        y_harmonic = librosa.effects.hpss(y)[0]

        spec_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_strength = np.sum(chroma, axis=0)

        spec_norm = (spec_centroid - np.mean(spec_centroid)) / (np.std(spec_centroid) + 1e-8)
        chroma_norm = (chroma_strength - np.mean(chroma_strength)) / (np.std(chroma_strength) + 1e-8)

        vocal_prob = (np.clip(spec_norm * 0.4, -1, 1) + np.clip(chroma_norm * 0.4, -1, 1)) / 2.0

        if len(vocal_prob) > 10:
            window_size = min(21, len(vocal_prob) // 5)
            if window_size >= 5:
                vocal_prob = signal.savgol_filter(vocal_prob, window_size | 1, 2)

        frame_times = librosa.frames_to_time(np.arange(len(vocal_prob)), sr=sr, hop_length=hop_length)

        vocal_threshold = 0.1
        vocal_frames = vocal_prob > vocal_threshold

        segments = []
        in_vocal = False
        start_time = 0.0

        for i, is_vocal in enumerate(vocal_frames):
            current_time = frame_times[i] if i < len(frame_times) else frame_times[-1]
            if is_vocal and not in_vocal:
                start_time = current_time
                in_vocal = True
            elif not is_vocal and in_vocal:
                if current_time - start_time > 1.0:
                    segments.append((start_time, current_time))
                in_vocal = False

        if in_vocal and len(frame_times) > 0:
            if frame_times[-1] - start_time > 1.0:
                segments.append((start_time, frame_times[-1]))

        logger.info(f"    Detected {len(segments)} vocal segments")
        return segments

    except Exception as e:
        logger.warning(f"    Vocal detection failed: {e}, using fallback")
        duration = len(y) / sr
        return [(duration * 0.2, duration * 0.4), (duration * 0.6, duration * 0.8)]


def detect_vocals_edges(y: np.ndarray, sr: int, config: AudioConfig) -> List[Tuple[float, float]]:
    """Detect vocals on intro/outro windows only (faster for long tracks)."""
    if len(y) < 2048:
        return []

    scan_samples = int(sr * config.vocal_scan_seconds)

    if len(y) <= scan_samples * 2:
        return detect_vocals(y, sr)

    intro_audio = y[:scan_samples]
    outro_audio = y[-scan_samples:]
    outro_offset = (len(y) - scan_samples) / sr

    intro_segments = detect_vocals(intro_audio, sr)
    outro_local = detect_vocals(outro_audio, sr)
    outro_segments = [(s + outro_offset, e + outro_offset) for s, e in outro_local]

    merged = intro_segments + outro_segments
    logger.info(f"    Detected {len(merged)} vocal segments (intro/outro scan)")
    return merged


def detect_intro_outro(y: np.ndarray, sr: int, beats: np.ndarray) -> Tuple[float, float]:
    """Detect intro and outro sections using RMS energy profile."""
    duration = len(y) / sr

    n_fft = max(2, min(2048, len(y)))
    hop_length = max(1, min(512, max(1, n_fft // 4)))
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    hop_time = hop_length / sr
    time_frames = np.arange(len(rms)) * hop_time

    energy_threshold = (np.max(rms) - np.min(rms)) * 0.05 + np.min(rms)
    above_threshold = rms > energy_threshold

    if np.any(above_threshold):
        intro_end = time_frames[np.argmax(above_threshold)]
        outro_start = time_frames[len(rms) - 1 - np.argmax(above_threshold[::-1])]
    else:
        intro_end = duration * 0.15
        outro_start = duration * 0.85

    if len(beats) > 32:
        beat_intro_end = beats[min(16, len(beats) // 4)]
        beat_outro_start = beats[max(-16, -len(beats) // 4)]
        intro_end = min(intro_end, beat_intro_end)
        outro_start = max(outro_start, beat_outro_start)

    return intro_end, outro_start
