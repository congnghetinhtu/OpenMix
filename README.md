<div align="center">

# 🎧 OpenMix

**Seamless DJ-style mixes from a folder of audio tracks**

Intelligent transitions · Tempo matching · Vocal-aware crossfading · Harmonic key handling

</div>

---

## ✨ Features

### 🎵 Smart Track Analysis

| Capability | How |
|---|---|
| **Tempo (BPM)** | `librosa` beat tracking |
| **Musical Key** | Chromagram profile → 12-bin key estimate |
| **Energy** | RMS mean + variation |
| **Timbre** | Spectral centroid, rolloff, bandwidth, ZCR |
| **Vocals** | Harmonic-percussive separation |
| **Structure** | Intro/outro detection from RMS envelope |

### 🔄 Intelligent Transitions

- **Equal-power crossfading** with quintic smoothstep curves
- **Tempo synchronization** — invisible / increasing / decreasing modes
- **Key correction** — pitch shift inside the crossfade window only
- **Vocal-aware ducking** — symmetric ducking prevents muddy overlap
- **Low-pass masking** — outgoing track fades into the blend
- **Beat alignment** — snapped to strong beats, avoiding vocal cut-through

### 🎛️ Professional Mixing

- **Compatibility scoring** for optimal track ordering
- **Greedy reordering** for smooth energy/flow progression
- **Per-track + final loudness normalization**
- **Soft limiting + TPDF dither** for clean output
- **Transition clips** and a **machine-readable CSV log**

---

## 🚀 Quick Start

### Installation

```bash
git clone <your-repo-url> && cd OpenMix
pip install -r requirements.txt
```

Requires **Python ≥ 3.10**.

### Mix your first playlist

```bash
python openmix.py /path/to/your/music/folder
```

That's it — OpenMix analyzes every track, picks the best order, and blends them together.

### Options

| Argument | Description | Default |
|---|---|---|
| `input_folder` | Path to folder containing audio tracks | **required** |
| `-o, --output` | Output filename | `openmix_output.wav` |
| `-s, --sample-rate` | Sample rate for processing | `44100` |

```bash
python openmix.py ~/Music/PartyMix -o "party_night_mix.wav" -s 48000
```

### 🎛️ Interactive Track Ordering

Run from a terminal and OpenMix asks for your preferred order:

```
Found 4 tracks:
  1. Track A.mp3
  2. Track B.flac
  3. Track C.mp3
  4. Track D.flac

Enter track order (e.g. 1 2 3 4)
or press Enter for auto-order:
```

Type an order (e.g. `3 1 2 4`) or press **Enter** for automatic ordering. Non-interactive contexts skip the prompt.

### 🐍 Programmatic API

```python
from cli import run

run("/path/to/music")                    # defaults
run("/path/to/music", "mix.wav", 48000)  # custom output + sample rate
```

Legacy wrapper:

```python
from openmix import OpenMixer

mixer = OpenMixer("/path/to/music", "mix.wav", 48000)
mixer.create_mix()
```

---

## 🧠 How It Works

### Step 1 — Audio Analysis

Each track is analyzed with `analyzer.py`:

- **Tempo (BPM)** → `librosa.beat.beat_track` for beat-matching and phase alignment
- **Musical Key** → chroma STFT averaged into a 12-bin profile; argmax = key
- **Energy** → RMS mean and variation
- **Spectral features** → centroid, rolloff, bandwidth, ZCR for timbral matching
- **Vocals** → harmonic-percussive separation on intro/outro windows
- **Intro/Outro** → RMS energy profile finds musical start & end

### Step 2 — Smart Ordering

`smart_track_ordering` greedily picks the next track with the highest compatibility score.

| Factor | Weight |
|---|---|
| Tempo similarity | **35%** |
| Key relationship | **30%** |
| Energy similarity | **20%** |
| Spectral similarity | **15%** |

Key scoring rewards **same key** (1.0), **fifths** (0.8), and **thirds** (0.7), with a linear falloff for other distances.

### Step 3 — Seamless Transitions

For each adjacent pair, `cli.py` orchestrates `mixer.py` + `crossfader.py`:

1. **Smooth flow** — skips the incoming track's intro vocals so vocals never collide
2. **Beat alignment** — snaps the transition point to strong beats on both sides
3. **Crossfade**:
   - **Tempo sync** — BPMs differ > 1.5 → time-stretch both sides (large diffs meet at midpoint)
   - **Key correction** — dissonant key distances → pitch shift ±2 semitones (window only)
   - **Low-pass masking** — progressive filtering, RMS-matched per segment
   - **Vocal ducking** — simultaneous energy → symmetric smoothed ducking
   - **Click prevention** — 256-sample boundary blend
4. **Final assembly** — RMS normalize → soft limit → TPDF dither → write

---

## 📦 Output

The script creates, in the input folder:

| Artifact | Description |
|---|---|
| `openmix_output.wav` | The full mixed file |
| `transitions/transition_1_2.wav` … | 30 s clips centered on each transition, for auditing |
| `openmix_output.csv` | Per-transition metadata: tempos, key diff, score, sync mode, ducking, position |

---

## 📁 Project Structure

```
OpenMix/
├── analyzer.py       # tempo, key, energy, spectral, vocal, intro/outro
├── audio_utils.py    # normalization, time-stretch, crossfade curves, limiting
├── cli.py            # pipeline orchestration + argparse entry point
├── crossfader.py     # per-transition crossfade engine
├── mixer.py          # compatibility, ordering, beat alignment, CSV logging
├── models.py         # AudioConfig, TrackAnalysis, TransitionLog, CrossfadeDebug
├── constants.py      # tuning constants
├── openmix.py        # legacy entry point / OpenMixer wrapper
├── tests/            # unit tests (pytest)
├── requirements.txt  # runtime dependencies
└── pyproject.toml    # packaging + mypy/ruff config
```

---

## 🎚️ Tuning Constants

Mixing behavior is tuned in `constants.py`:

| Constant | Default | Purpose |
|---|---|---|
| `BLEND_SAMPLES` | 256 | Boundary blend to prevent clicks |
| `NORMALIZE_TARGET_RMS` | 0.15 | Target loudness for normalization |
| `SOFT_LIMIT_CEILING` | 0.95 | Peak ceiling after limiting |
| `TEMPO_RANGE_BPM` | 30 | Tempo score falloff range |
| `SPECTRAL_CENTROID_RANGE` | 2000 | Spectral score falloff range |
| `MIN_BEATS_FOR_STRONG` | 4 | Minimum beats to count as "strong" |
| `STRIDE_STRONG_BEATS` | 4 | Beat stride for strong-beat alignment |
| `TEMPO_MAX_STRETCH_*` | .02–.15 | Graduated stretch caps by BPM diff |

> The crossfade duration is fixed at **15.0 s** in `cli.run()` (also the default in `models.AudioConfig` / `Crossfader`).

---

## 💡 Tips for Best Results

<details>
<summary><b>🎵 Track Selection</b></summary>

- Similar BPM ranges work best (±20 BPM)
- Same or related keys (fifth, third) → smoother transitions
- Plan energy progression: start calm → build → wind down

</details>

<details>
<summary><b>🖥️ Performance</b></summary>

- Use WAV/FLAC for best quality
- Limit to 10–15 tracks per mix for reasonable processing time
- Close heavy applications; use a lower sample rate when testing

</details>

<details>
<summary><b>🔧 Troubleshooting</b></summary>

| Issue | Fix |
|---|---|
| "No audio files found" | Folder must contain supported formats, path must exist |
| "Need at least 2 tracks" | Add at least two audio files |
| "Error analyzing [file]" | File corrupted or unsupported — try another file |
| Memory issues | Smaller batches, lower sample rate |

</details>

---

## ✅ Tests

```bash
python -m pytest tests/
```

Covers data-model construction, crossfade curve endpoints, soft limiting, and audio normalization.

---

## 🧪 Verify with a real mix

```bash
python openmix.py tracks/
```

The repo ships a sample `tracks/` folder — inspect the generated `transitions/*.wav` clips and the CSV log to see the algorithm's decisions.

---

## 📜 License

**GPL-3.0** — see `LICENSE`. Ensure you have rights to the audio files you're processing.

---

<div align="center">

*Enjoy creating professional-quality mixes with the power of Python! 🎧*

</div>
