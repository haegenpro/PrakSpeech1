# Metodologi Tugas 3: Model Gaussian

```text
Pelatihan: rekaman 01 -> MFCC 39 -> potong hening -> kumpulkan frame per vokal -> mean + kovarians
Prediksi: input -> potong hening -> NLL rata-rata per frame terhadap 5 model -> NLL terkecil = vokal
```

| Komponen | Pilihan |
|---|---|
| Potong hening | Sama dengan DTW (`src/silence.py`, 20 dB dari frame terkeras) |
| Model | 1 Gaussian per vokal, 39 dimensi, dilatih pada frame (bukan rekaman utuh) |
| Data | Hanya partisi template (5 speaker, sekitar 220-260 frame per vokal), tidak ada data uji |
| Mean | Rata-rata frame |
| Kovarians | Kovarians frame penuh, dikecilkan ke diagonal: `S' = (1-a)S + a diag(S)` |
| Lantai | `S' += 1e-6 * rata-rata diagonal * I` agar selalu definit positif |
| Biaya | NLL rata-rata per frame, `0.5 (d log 2pi + log det S' + (x-mu)' S'^-1 (x-mu))`, dihitung dengan Cholesky |
| Fitur | 39 dimensi penuh, tanpa CMN, sama dengan DTW |

`a = 0` kovarians penuh, `a = 1` diagonal. Kovarians penuh dari sekitar 250 frame yang saling berkorelasi
kondisinya buruk (cond 3e5-8e5), jadi diperlukan pengecilan. Pengecilan ke diagonal dipilih daripada `lambda * I`
karena c0 dan delta-delta berbeda skala beberapa orde besar.

Parameter dipilih dengan leave-one-speaker-out pada partisi template (protokol sama dengan DTW):

| Variasi | Benar (dari 25) |
|---|---|
| Tanpa potong, a=0 | 14 |
| Tanpa potong, a=0.01 | 14 |
| Tanpa potong, a=0.1 | 15 |
| Tanpa potong, a=0.5 | 16 |
| Tanpa potong, a=1 | 12 |
| Potong 20 dB, a=0 | 22 |
| Potong 20 dB, a=0.01 | 22 |
| **Potong 20 dB, a=0.1 (dipakai)** | **22** |
| Potong 20 dB, a=0.5 | 20 |
| Potong 20 dB, a=1 | 16 |

a = 0, 0.01, 0.1 seri (22 dari 25), dipilih a = 0.1 karena pengecilan terbesar di antara yang seri.
Sebagai pembanding, DTW mendapat 20 dari 25 dengan protokol yang sama. Data closed dan open tidak dipakai
untuk memilih parameter.

| Perintah | Fungsi |
|---|---|
| `python scripts/build_gaussian.py` | Buat `models/gaussian_models.npz` (`--alpha` untuk mengubah a) |
| `python scripts/predict_gaussian.py file.wav` | Prediksi satu file |
| `python scripts/tune_gaussian.py` | Ulangi tabel leave-one-speaker-out di atas |
| `predict(item["features"], load_models(...))` | Dipakai Person 4, antarmuka sama dengan `dtw_predict.predict` |
