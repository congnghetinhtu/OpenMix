"""
OpenMix data models.
Maps directly to Swift structs when porting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    crossfade_duration: float = 15.0
    max_analysis_seconds: int = 15
    vocal_scan_seconds: int = 10
    supported_formats: tuple = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg')


@dataclass
class TrackAnalysis:
    file_path: Path
    duration: float
    tempo: float
    beats: np.ndarray
    beat_frames: np.ndarray
    key: int
    energy: float
    energy_variation: float
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    zcr: float
    vocal_segments: List[Tuple[float, float]]
    intro_end: float
    outro_start: float
    peak_level: float
    rms_level: float
    audio_data: np.ndarray
    sample_rate: int

    def __post_init__(self):
        self.file_path = Path(self.file_path)


@dataclass
class TransitionLog:
    transition: int
    from_track: str
    to_track: str
    from_tempo: float
    to_tempo: float
    tempo_diff: float
    tempo_sync_mode: str
    from_key: int
    to_key: int
    key_diff: int
    compatibility_score: float
    crossfade_sec: float
    vocal_segments_outgoing: int
    vocal_segments_incoming: int
    ducking_applied: bool
    ducking_frames: int
    mix_position_sec: float
    phase_correlation: float = 0.0
    zero_crossing_aligned: bool = False
    phase_inverted: bool = False

    def to_dict(self) -> dict:
        return {
            'transition': self.transition,
            'from_track': self.from_track,
            'to_track': self.to_track,
            'from_tempo': self.from_tempo,
            'to_tempo': self.to_tempo,
            'tempo_diff': self.tempo_diff,
            'tempo_sync_mode': self.tempo_sync_mode,
            'from_key': self.from_key,
            'to_key': self.to_key,
            'key_diff': self.key_diff,
            'compatibility_score': self.compatibility_score,
            'crossfade_sec': self.crossfade_sec,
            'vocal_segments_outgoing': self.vocal_segments_outgoing,
            'vocal_segments_incoming': self.vocal_segments_incoming,
            'ducking_applied': self.ducking_applied,
            'ducking_frames': self.ducking_frames,
            'mix_position_sec': self.mix_position_sec,
            'phase_correlation': self.phase_correlation,
            'zero_crossing_aligned': self.zero_crossing_aligned,
            'phase_inverted': self.phase_inverted,
        }


@dataclass
class CrossfadeDebug:
    tempo_sync_mode: str = 'none'
    ducking_applied: bool = False
    ducking_frames: int = 0
    crossfade_section: Optional[np.ndarray] = None
    phase_correlation: float = 0.0
    zero_crossing_aligned: bool = False
    phase_inverted: bool = False
