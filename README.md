# PrakSpeech1 — Task 1: Dataset and MFCC Features

## 1. Project overview

The practicum **Pengenalan Vowel Bahasa Indonesia dengan DTW dan Model Gaussian**
recognizes `/a/`, `/i/`, `/u/`, `/e/`, `/o/`. It requires Python, two recordings per
vowel per internal speaker, and exactly 39 values per speech frame.

| Canonical ID | Group member | Original filename aliases |
|---|---|---|
| speaker01 / S01 | Peter Wongsoredjo | Peter, S01 |
| speaker02 / S02 | Jessica Allen | Allen |
| speaker03 / S03 | Haegen Quinston | Haegen |
| speaker04 / S04 | Nathaniel JR | Nathan, Nathaniel |
| speaker05 / S05 | Maheswara Bayu Kaindra | Indra |

## 2. Task 1 scope

Implemented: canonical dataset import, WAV validation and preprocessing,
39-dimensional frame features, metadata/provenance, partition loading, CLI tools,
and automated tests. Recognition, averaged templates, Gaussian parameter estimation,
and accuracy evaluation are assigned to other modules and are not implemented here.

## 3. Dataset requirements

Five speakers × five vowels × two takes = **50 internal recordings**. Each speaker
contributes five template and five closed-speaker test recordings. External speakers
are separate identities and have no fixed required recording count.

Preserved `data/train/<vowel>/` recordings map take 1 to template and take 2 to
closed-speaker testing. Preserved `data/test/<vowel>/G07_SXX_V.wav` recordings map
to open-speaker testing, as confirmed by the owner. Their identities become
`external_g07_sXX`, distinct from internal IDs. `data/import_manifest.json` records
each explicit source/canonical mapping, aliases, and SHA256.

Canonical filenames: `speaker01_a_01.wav` (template), `speaker01_a_02.wav` (closed),
and `external_g07_s01_a_01.wav` (open). External IDs start with `external_`; vowel
labels are lowercase. Add future data directly to canonical directories. Extend
the explicit importer mapping for other legacy naming conventions.

## 4. Recording instructions

Record one natural, sustained vowel per WAV in a quiet room at a consistent
microphone distance and gain. Leave a small margin around the vowel and avoid
clipping. Record two distinct utterances; do not duplicate a take. Prefer mono
PCM16 WAV at 16 kHz; other decodable WAV rates and channel counts are supported.
Listen to verify pronunciation and noise. Never replace missing speech with
synthetic signals.

## 5. Folder organization

```text
PrakSpeech1/
├── data/
│   ├── train/<vowel>/                 # preserved internal originals
│   ├── template/speaker01..05/         # canonical take 01
│   ├── test/
│   │   ├── <vowel>/                    # preserved G07 originals
│   │   ├── closed/speaker01..05/       # canonical take 02
│   │   └── open/external_g07_s01..05/  # external recordings
│   └── import_manifest.json
├── features/
│   ├── template/<speaker>/*.npy
│   ├── test/closed/<speaker>/*.npy
│   ├── test/open/<speaker>/*.npy
│   ├── metadata.csv
│   └── manifest.json
├── src/{__init__,preprocessing,dataset_utils}.py
├── scripts/{setup_dataset,validate_dataset,extract_features}.py
├── tests/{test_preprocessing,test_dataset}.py
├── report/{task1_methodology.md,task1_validation.json}
├── requirements.txt
└── pytest.ini
```

Empty canonical directories contain `.gitkeep`; `external_speaker01` is an empty
placeholder for manually supplied data. Setup is idempotent, checks collisions
before copying audio, and rejects conflicting destinations. Originals are preserved.

## 6. Installation instructions

Use Python 3.11 or newer. Tested here with Python 3.14, NumPy 2.3.3, SciPy 1.16.3,
SoundFile 0.13.1, python_speech_features 0.6, and pytest 9.1.1.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

SoundFile Windows wheels include libsndfile. Platforms without a suitable wheel
may require a system libsndfile installation. No recognition framework is required.

## 7. Audio preprocessing methodology

`load_audio(path)` returns finite float32 mono samples and the **original** sample
rate. Stereo channels are averaged. `preprocess_audio(audio, rate, config)` resamples
using SciPy polyphase filtering, rather than changing sample-rate metadata.
`process_audio_file(path, config)` runs the full pipeline.

| Setting | Default | Purpose / impact |
|---|---|---|
| Target rate | 16,000 Hz | Consistent frequency interpretation |
| Channels / dtype | mono / float32 | Common downstream representation |
| Edge trimming | off | Preserve vowel onset and offset |
| Peak normalization | off | Preserve energy differences |
| Pre-emphasis | 0.97 | Spectral tilt compensation, once inside MFCC |
| Frame / step | 25 / 10 ms | Overlapping short-time analysis |
| Window | Hamming | Reduce spectral leakage |
| Filters / FFT | 26 / 512 | Mel spectral analysis |
| Delta radius | 2 frames | Local derivative estimate |

These are implementation choices; the practicum mandates 39 dimensions but not
these numeric defaults. Consistent settings across partitions and inference are
necessary. Trimming cuts only low-amplitude edges using a relative peak threshold
(default 0.01); it is optional and is not voice activity detection. Normalization
divides by peak amplitude and changes energy features. Neither option is required.
Empty, corrupt, nonfinite, and all-zero inputs are rejected. Opposite-phase stereo
can cancel during averaging and is rejected if the resulting waveform is silent.
No cepstral normalization or learned preprocessing is applied.

## 8. MFCC 39-dimensional feature extraction

The selected decomposition is **13 static + 13 delta + 13 delta-delta**. The
practicum requires 39 dimensions but does not mandate this decomposition.

The permitted `python_speech_features` library applies pre-emphasis
`y[n] = x[n] - 0.97*x[n-1]`, framing/windowing, FFT power, Mel filtering, log energies,
and a type-II DCT. It uses cepstral lifter 22; coefficient 0 is replaced by log
frame energy (`appendEnergy=True`). Delta coefficients are calculated as:

```text
delta_t = sum(n * (c_(t+n) - c_(t-n)), n=1..N) / (2 * sum(n², n=1..N))
N = 2; delta-delta applies the same operation to delta.
```

Boundary frames are replicated. Inputs shorter than one frame are zero-padded;
the library also pads the last partial frame. A one-frame sequence has zero
derivatives; numerical validity does not establish recognition quality.
Output is float32 `(T, 39)`, ordered as `[:, :13]`, `[:, 13:26]`, `[:, 26:39]`.
Time order and variable T are preserved. A frame is a short window; a vowel
recording contains an entire utterance and typically many frames.

## 9. Execution commands

Run from the repository root:

```powershell
python scripts/setup_dataset.py
python scripts/validate_dataset.py --report report/task1_validation.json
python scripts/extract_features.py --data-dir data --output-dir features
python -m pytest -q
```

```powershell
# Optional settings; all partitions must use the same configuration.
python scripts/extract_features.py --trim-silence --normalize --overwrite
python scripts/extract_features.py --help
```

Change `--nfft` when choosing frames too long for 512 points. Restore defaults
with a default extraction plus `--overwrite`. Changed sources, settings or
dependency versions require `--overwrite` for incompatible existing files.
Extraction reports processed/reused/failed/errors and exits 1 on failures.
Missing recordings are reported by validation; extraction supports partial data.
Validation exits 1 for missing/invalid/duplicate recordings. Mixed supported
audio formats are reported but can be normalized through preprocessing.

## 10. Output files and feature formats

Example: `features/template/speaker01/speaker01_a_01.npy`. Use `load_feature_file`
or NumPy with `allow_pickle=False` to load a sequence.

`metadata.csv` stores canonical/source paths, filename, speaker, vowel, partition,
original rate/channels/samples/duration, processed rate/samples/duration, frame
count, feature dimensions/path, status, and errors. Processed duration precedes
frame padding and follows optional trimming. Canonically named bad audio gets an
error row; invalid names appear in CLI diagnostics and the validation report.

`manifest.json` stores configuration, pipeline/dependency versions, data root,
source hashes and feature hashes. Each file replacement is atomic. An interrupted
multi-file run may need re-extraction; loaders reject index/hash mismatches.
Unindexed `.npy` files are ignored. Outputs are reused only when provenance matches.
Generated source paths are absolute; rerun extraction after moving the project.
Canonical WAVs must remain available for loader provenance checks.

## 11. Integration instructions for DTW and Gaussian

```python
from src.preprocessing import process_audio_file, load_feature_file
from src.dataset_utils import load_dataset, VOWELS

features = process_audio_file("data/template/speaker01/speaker01_a_01.wav")
print(features.shape)  # (T, 39); T depends on the recording

template_data = load_dataset("template")
closed_data = load_dataset("closed")
open_data = load_dataset("open")
for item in template_data:
    sequence = item["features"]  # full (T, 39) sequence for DTW
    label, speaker, wav_path = item["vowel"], item["speaker"], item["file_path"]
```

The return type is a sorted list of dictionaries. An empty indexed partition
returns `[]`. Unknown partitions and stale/missing indexed files raise errors;
failed rows are excluded. For Gaussian frame grouping:

```python
import numpy as np
frames_by_vowel = {
    vowel: np.concatenate([r["features"] for r in template_data if r["vowel"] == vowel])
    for vowel in VOWELS if any(r["vowel"] == vowel for r in template_data)
}
```

This estimates no parameters. Do not mix test frames into training. The practicum's
**closed-recording** scenario reuses template recordings as evaluation inputs;
**closed-speaker** uses different takes from the same speakers; **open-speaker**
uses G07 speakers. Closed-recording needs no additional copied partition.
For custom settings, pass the same `PreprocessingConfig` to extraction and
`process_audio_file` during inference; consult the saved manifest.

## 12. Testing and validation

Tests cover all ten requested scenarios plus short/invalid audio, coefficient
ordering, single pre-emphasis, aliases, import conflicts/idempotence, duplicate
audio, cache reuse, incompatible configuration, tampering, and removed sources.
Synthetic tones are confined to temporary pytest fixtures. Pytest uses the
disposable `.pytest_cache/task1-temp` folder because the system temp location
had an access error on this Windows installation.

Measured on **25 October 2026 (WIB)**:

| Check | Observed result |
|---|---|
| Internal dataset | 25 template + 25 closed; all 50 expected files present |
| External dataset | 25 open recordings |
| Decoding | 75 successful; no invalid names or exact audio duplicates |
| Format | All 75 mono 16 kHz PCM16 WAV |
| Extraction | 75 files, 0 failures |
| Repeated extraction | 75 compatible files reused |
| Frames | 70–495 per recording; 15,564 total |
| Integration loader | 25 records per partition |

The saved JSON records per-file validation; current pytest output reports test
results. Rerun checks after changing recordings or configuration.

## 13. Known limitations

Structural checks cannot establish pronunciation, speaker authenticity, clipping
acceptability, or sufficient signal-to-noise ratio. G07 provenance follows owner
confirmation. Duplicate detection compares decoded mono samples and rate exactly;
gain-adjusted copies and perceptually similar recordings are not detected.
Preserved legacy sources are excluded from canonical discovery. Optional trimming
is not a denoiser. Concurrent extraction into one output directory is unsupported.
Recognition accuracy remains the responsibility of the evaluation modules.
