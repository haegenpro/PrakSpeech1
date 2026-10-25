# Metodologi Tugas 3: Model Gaussian

```text
Pelatihan: rekaman 01 -> MFCC 39 -> potong hening -> kumpulkan frame per vokal -> mean + kovarians
Prediksi: input -> potong hening -> NLL rata-rata per frame terhadap 5 model -> NLL terkecil = vokal
```

| Komponen | Pilihan |
|---|---|
| Potong hening | Sama dengan DTW (`src/silence.py`, 20 dB dari frame terkeras) |
| Model | 1 Gaussian per vokal (K = 1 segmen), 39 dimensi, dilatih pada frame (bukan rekaman utuh) |
| Data | Hanya partisi template (5 speaker, sekitar 220-260 frame per vokal), tidak ada data uji |
| Mean | Rata-rata frame |
| Kovarians | Kovarians frame penuh 39 x 39 (pembagi N-1) |
| Lantai | `C += 1e-6 * rata-rata diagonal * I` agar selalu definit positif |
| Biaya | NLL rata-rata per frame, `0.5 (d log 2pi + log det C + (x-mu)' C^-1 (x-mu))`, dihitung dengan Cholesky |
| Fitur | 39 dimensi penuh, tanpa CMN, sama dengan DTW |

Model ini adalah kasus K = 1 dari generalized template pada materi kuliah: satu segmen per vokal, sehingga
segmental K-means dan Viterbi tidak diperlukan. Vokal yang ditahan relatif stasioner. Percobaan awal dengan
K = 2-4 segmen (leave-one-speaker-out, hanya partisi template) memberi 19-20 dari 25, tidak lebih baik dari K = 1 (22).

Penyusutan kovarians ke diagonal (`C' = (1-a)C + a diag(C)`) sempat diuji dan dibuang karena tidak memperbaiki
hasil: pada leave-one-speaker-out, a = 0 (kovarians penuh), 0.01 dan 0.1 sama-sama 22 dari 25, a = 0.5 turun ke 20
dan a = 1 (diagonal) ke 16. Kovarians penuh dari sekitar 250 frame masih cukup terkondisi (cond 3e5-7e5).

Pemotongan hening dipilih dengan leave-one-speaker-out pada partisi template (protokol sama dengan DTW):

| Variasi | Benar (dari 25) |
|---|---|
| Tanpa potong | 14 |
| **Potong 20 dB (dipakai)** | **22** |

Sebagai pembanding, DTW mendapat 20 dari 25 dengan protokol yang sama. Data closed dan open tidak dipakai
untuk memilih parameter.

| Perintah | Fungsi |
|---|---|
| `python scripts/build_gaussian.py` | Buat `models/gaussian_models.npz` |
| `python scripts/predict_gaussian.py file.wav` | Prediksi satu file |
| `python scripts/tune_gaussian.py` | Ulangi tabel leave-one-speaker-out di atas |
| `predict(item["features"], load_models(...))` | Dipakai Person 4, antarmuka sama dengan `dtw_predict.predict` |
