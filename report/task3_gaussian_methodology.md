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
segmental K-means dan Viterbi tidak diperlukan. Vokal yang ditahan relatif stasioner, jadi urutan frame tidak
dipakai dan panjang rekaman tidak perlu diselaraskan.

Pemotongan hening dipakai karena rekaman berdurasi sekitar 4-5 detik sedangkan vokalnya hanya sekitar 0,3 detik;
tanpa pemotongan, frame hening mendominasi mean dan kovarians. Parameter yang sama dengan DTW dipakai agar
perbandingan adil. Data closed dan open tidak
dipakai untuk memilih parameter apa pun.

| Perintah | Fungsi |
|---|---|
| `python scripts/build_gaussian.py` | Buat `models/gaussian_models.npz` |
| `python scripts/predict_gaussian.py file.wav` | Prediksi satu file |
| `predict(item["features"], load_models(...))` | Dipakai Person 4, antarmuka sama dengan `dtw_predict.predict` |
