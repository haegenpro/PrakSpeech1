# Metodologi Tugas 2: DTW Averaged Template

```text
Template: rekaman 01 -> MFCC 39 -> potong hening -> medoid -> selaraskan DTW -> rata-rata
Prediksi: input -> potong hening -> DTW ke 5 template -> biaya terkecil = vokal
```

| Komponen | Pilihan |
|---|---|
| Potong hening | Frame pertama sampai terakhir dengan energi (c0) dalam 20 dB dari frame terkeras |
| Jarak lokal | Euclidean 39 dimensi |
| Langkah DTW | (i-1, j-1), (i-1, j), (i, j-1) |
| Biaya | Biaya akumulasi / panjang jalur |
| Template | Medoid per vokal, 5 sekuens diselaraskan lalu dirata-rata |
| Data | Hanya partisi template, tidak ada data uji |

Parameter dipilih dengan leave-one-speaker-out pada partisi template:

| Variasi | Benar (dari 25) |
|---|---|
| Tanpa potong hening | 12 |
| Potong 30 dB | 16 |
| **Potong 20 dB (dipakai)** | **20** |
| 20 dB + z-score | 20 |
| 20 dB + 3 iterasi rata-rata | 20 |

| Perintah | Fungsi |
|---|---|
| `python scripts/build_templates.py` | Buat `models/dtw_templates.npz` |
| `python scripts/predict_dtw.py file.wav` | Prediksi satu file |
| `predict(item["features"], load_templates(...))` | Dipakai Person 4 |
| `trim_silence(features)` | Bisa dipakai Person 3 |
