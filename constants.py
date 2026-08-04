"""OpenMix tuning constants. Edit these to adjust mixing behavior."""

# Crossfader
BLEND_SAMPLES = 256
FILTER_ORDER = 4
FILTER_MIN_HZ = 200.0
FILTER_SEARCH_RANGE = 0.1

# Normalization
NORMALIZE_TARGET_RMS = 0.15
NORMALIZE_GAIN_MIN = 0.3
NORMALIZE_GAIN_MAX = 2.0
FINAL_GAIN_MIN = 0.8
FINAL_GAIN_MAX = 1.2
SOFT_LIMIT_CEILING = 0.95

# Compatibility scoring
TEMPO_RANGE_BPM = 30
SPECTRAL_CENTROID_RANGE = 2000

# Analysis
MIN_BEATS_FOR_STRONG = 4
STRIDE_STRONG_BEATS = 4

# Tempo sync (graduated stretch caps)
TEMPO_MAX_STRETCH_INVISIBLE = 0.02   # < 4 BPM diff
TEMPO_MAX_STRETCH_SMALL = 0.05       # 4-10 BPM diff
TEMPO_MAX_STRETCH_LARGE = 0.15       # > 10 BPM diff
TEMPO_STRETCH_RAMP_DURATION = 0.5    # fraction of crossfade for ramp-up
