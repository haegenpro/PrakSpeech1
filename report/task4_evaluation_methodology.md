# Metodologi Tugas 4: Evaluasi

Pelatihan hanya memakai partisi `template` (take 01, 5 speaker x 5 vowel). Parameter DTW dan Gaussian tidak diubah
setelah melihat data uji.

| Skenario | Data uji | Jumlah |
|---|---|---|
| Closed recording | Partisi `template` (rekaman yang sama dengan template) | 25 |
| Closed speakers | Partisi `closed` (speaker sama, take 02) | 25 |
| Open speakers | Partisi `open` (external_g07_s01..05) | 25 |

Akurasi = prediksi benar / jumlah data uji x 100%. Rata-rata = rata-rata tiga skenario (masing-masing 25 data, jadi sama dengan akurasi gabungan 75 data).

| Perintah | Fungsi |
|---|---|
| `python scripts/evaluate.py` | Jalankan semua prediksi, tulis `results/*.csv` dan `results/evaluation_report.md` |
| `python scripts/evaluate.py --reuse` | Bangun ulang tabel dari `results/predictions.csv` |

Keluaran: `predictions.csv` (satu baris per sistem x data uji, dengan biaya ke 5 vowel), `table0_summary.csv`,
`table1_per_vowel.csv`, `table2_closed_per_speaker.csv`, `table3_open_per_speaker.csv`, confusion matrix dan daftar kesalahan.