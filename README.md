# PrakSpeech1: Indonesian vowel recognition

This practicum compares averaged-template DTW and Gaussian recognition for
`/a/`, `/i/`, `/u/`, `/e/`, and `/o/`. Task 1 prepares the dataset and extracts
39-dimensional features for each speech frame. Recognition and evaluation live
in the other modules.

## Dataset

| Internal ID | Member | Legacy filename aliases |
|---|---|---|
| speaker01 | Peter Wongsoredjo | Peter, S01 |
| speaker02 | Jessica Allen | Allen |
| speaker03 | Haegen Quinston | Haegen |
| speaker04 | Nathaniel JR | Nathan, Nathaniel |
| speaker05 | Maheswara Bayu Kaindra | Indra |

Each member records every vowel twice: take 01 supplies the template/training
partition and take 02 supplies the closed-speaker test partition. Closed-recording
evaluation reuses the template recordings.

The current open partition contains 80 recordings from 16 external speakers:
G03/Kel3 (S01, S02, S03, S05), G04 (S03, S04), G07 (S01–S05), and G09 (S01–S05).
Each provides five vowels. Group-aware IDs such as `external_g03_s01` keep people
from different groups separate. External data is used only for testing.

```text
data/
  template/<speaker>/speaker01_a_01.wav
  test/closed_speakers/<speaker>/speaker01_a_02.wav
  test/open_speakers/<external_id>/external_g03_s01_a_01.wav
features/
  template/<speaker>/*.npy
  test/closed_speakers/<speaker>/*.npy
  test/open_speakers/<external_id>/*.npy
  metadata.csv
```

Record distinct utterances in a quiet room, with a small margin around each vowel
and consistent microphone distance and gain. Prefer mono PCM16 WAV at 16 kHz.
Listen to check pronunciation, clipping, and background noise.

## Run Task 1

Use Python 3.11 or newer and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/validate_dataset.py --report report/task1_validation.json
python scripts/extract_features.py
python -m pytest -q
```

Extraction accepts only `--data-dir` and `--output-dir`. Every run regenerates
all discovered canonical features and replaces the metadata index. Run it after
changing recordings or the shared settings. Failed files receive error rows;
the command reports processed/failed counts and exits nonzero on failures.
Validation checks completeness separately, so partial datasets can be extracted.

Existing canonical data needs no setup run. For legacy uploads, run
`python scripts/setup_dataset.py`. It maps internal aliases and external names
such as `Kel3_S01_A.wav`, `G04_S03_A.wav`, and `G09_S01_A01.wav` into canonical
folders. All mappings and byte-level collisions are checked before copying.
Originals are preserved, identical destinations are skipped, and conflicting
destinations raise errors. Old `test/closed` and `test/open` folders can also be
copied to the current directory names. No import manifest is needed.

## Features and preprocessing

`src/preprocessing.py` defines `DEFAULT_CONFIG`, shared by extraction and inference.
The Python functions still accept a `PreprocessingConfig` for controlled experiments.

| Setting | Default |
|---|---|
| Sample rate and channels | 16 kHz, mono (channel average) |
| Frame length / step | 25 ms / 10 ms |
| Window | Hamming |
| Mel filters / FFT size | 26 / 512 |
| Pre-emphasis | 0.97, applied once inside MFCC |
| Cepstral lifter / energy | 22 / first coefficient is log frame energy |
| Temporal derivatives | Delta radius 2; delta-delta from delta |
| Waveform edge trimming / peak normalization | Both off |

Each output is a finite float32 array shaped `(T, 39)`: 13 static MFCCs, 13 deltas,
and 13 delta-deltas. Time order is preserved. Short audio is zero-padded; derivative
boundaries replicate the edge frames. The specification requires 39 values per
frame; the decomposition and numerical settings are implementation choices.

WAV loading rejects empty, corrupt, nonfinite, and all-zero audio. SciPy polyphase
resampling changes the samples to the target rate. Optional waveform trimming
uses a relative peak-amplitude threshold; normalization divides by peak amplitude.
These options are set in the shared configuration, not on the extraction CLI.

Separately, DTW and Gaussian trim feature-frame edges using log frame energy
(20 dB below the peak) by default. This happens after MFCC extraction and does not
recompute derivatives at the retained boundaries. It is different from optional
waveform trimming. Apply the same policy to training and recognition inputs.

## Loading and integration

```python
from src.dataset_utils import load_dataset
from src.preprocessing import process_audio_file

template = load_dataset("template")
closed = load_dataset("closed")
opened = load_dataset("open")
features = process_audio_file("data/template/speaker01/speaker01_a_01.wav")
for recording in template:
    sequence = recording["features"]  # (T, 39)
    label = recording["vowel"]
    speaker = recording["speaker"]
    wav_identifier = recording["file_path"]
```

`load_dataset(partition, features_dir="features")` reads `metadata.csv` and validates
feature arrays, identity, partition, and contained relative paths. Missing or
invalid indexed features raise errors; failed rows and unindexed arrays are ignored.
Original WAVs and manifests are not required. Returned WAV identifiers resolve
against the `data` directory beside the selected feature directory; these are
identifiers and do not need to exist. An empty indexed partition returns `[]`.

Metadata retains labels, canonical relative WAV/source paths, original and processed
audio properties, frame counts, feature paths, and validation status. There is no
cache, hash enforcement, or dependency-version gate. Always regenerate features
after editing audio or preprocessing settings. Use the same configuration for live
inference; use only template frames for Gaussian training.

## Validation and measured results

Explicit validation checks the 50 expected internal recordings, canonical names,
decoding, exact decoded-audio duplicates, and five-vowel coverage for each discovered
external speaker. External speaker counts and group IDs are not fixed in code.
These checks cannot establish pronunciation or speaker authenticity.

On 25 October 2026 (WIB), the dataset contains 25 template, 25 closed-speaker, and
80 open-speaker recordings from 16 external speakers. All 130 decode as mono
16 kHz PCM16, with no missing internal files, missing external vowels, invalid names,
or exact duplicates. Features span 33–495 frames, totaling 22,147 frames.

The simplification regenerated all 130 arrays with exactly the same shapes and
values as before. It does not change MFCC mathematics or recognition behavior.
Models need no rebuild solely for this refactor; results predating the expanded
open dataset must be reevaluated on all 80 open recordings.

Tests cover preprocessing, canonical identities, several external naming conventions,
import collisions, regeneration, invalid feature files, and loading without WAVs.
Synthetic tones are confined to temporary test directories.

Dependency versions are informational: Python 3.14, NumPy 2.3.3, SciPy 1.16.3,
SoundFile 0.13.1, python_speech_features 0.6, and pytest 9.1.1 were used for the
earlier Task 1 baseline. Installation ranges are in `requirements.txt`; changing a
version does not block loading or extraction. Concurrent extraction into a shared
output directory is unsupported.
