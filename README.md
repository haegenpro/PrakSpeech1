# Indonesian Vowel Recognition

Speech processing practicum comparing averaged-template Dynamic Time Warping
(DTW) and multivariate Gaussian models for five Indonesian vowels: /a/, /i/, /u/,
/e/, and /o/.

## Setup

Requires Python 3.11 or newer. Run commands from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

| Partition | Speakers | Recordings |
|---|---|---:|
| Template | 5 internal, take 01 | 25 |
| Closed speakers | Same 5 internal, take 02 | 25 |
| Open speakers | 16 external, from G03/Kel3, G04, G07, and G09 | 80 |

Closed-recording evaluation reuses the template recordings. External recordings
are used only for testing.

```text
data/
  template/<speaker>/*.wav
  test/closed_speakers/<speaker>/*.wav
  test/open_speakers/<external_id>/*.wav
features/                 # MFCC arrays and metadata.csv
models/                   # Saved DTW templates and Gaussian parameters
results/                  # Predictions, accuracy tables, and evaluation report
notebook/                 # Integration analysis and plots
report/                   # Methodology and dataset validation
```

## Run

```powershell
python scripts/validate_dataset.py --report report/task1_validation.json
python scripts/extract_features.py
python scripts/build_templates.py
python scripts/build_gaussian.py
python scripts/evaluate.py
python -m pytest -q
```

Resolve validation errors before continuing. Extraction regenerates all feature
files and metadata. Rerun it after changing audio or preprocessing settings.
Evaluation builds models from the template partition in memory and writes results
to `results/`; the build commands save models for individual predictions.

```powershell
python scripts/predict_dtw.py data/test/closed_speakers/speaker01/speaker01_a_02.wav
python scripts/predict_gaussian.py data/test/closed_speakers/speaker01/speaker01_a_02.wav
```

For legacy recordings, `python scripts/setup_dataset.py` copies supported filenames
into the dataset layout after checking for conflicts. Existing canonical data
needs no setup step.

## Methods

Audio is converted to mono and resampled to 16 kHz when necessary. Features use
25 ms Hamming windows, a 10 ms step, a 512-point FFT, 26 Mel filters, and
pre-emphasis of 0.97. Each frame contains 13 static coefficients, 13 deltas, and
13 delta-deltas, stored as a float32 array of shape `(T, 39)`. The first static
coefficient is log frame energy.

Waveform trimming and peak normalization are disabled by default. Both recognition
systems trim feature-frame edges at 20 dB below peak frame energy. DTW aligns and
averages training sequences per vowel; Gaussian models estimate a mean vector and
full covariance matrix with diagonal regularization.

Shared preprocessing settings are defined in `src/preprocessing.py`. Methodology
is documented in `report/`.

## Notebook

Install the additional analysis dependencies:

```powershell
python -m pip install pandas matplotlib seaborn jupyter
jupyter notebook
```

Open `notebook/integration_analysis.ipynb` and use the repository root as the kernel
working directory. Run the pipeline above first to generate features and saved
models. The notebook displays evaluation tables, confusion matrices, and timing
measurements.
