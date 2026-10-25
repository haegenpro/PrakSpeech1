# Metodologi Tugas 4: Evaluasi

Pelatihan hanya memakai partisi `template` (take 01, 5 speaker x 5 vowel). Parameter DTW dan Gaussian tidak diubah
setelah melihat data uji.

| Skenario | Data uji | Jumlah |
|---|---|---|
| Closed recording | Partisi `template` (rekaman yang sama dengan template) | 25 |
| Closed speakers | Partisi `closed` (speaker sama, take 02) | 25 |
| Open speakers | Partisi `open` (16 speaker eksternal dari G03, G04, G07, G09) | 80 |

Akurasi = prediksi benar / jumlah data uji x 100%. Rata-rata pada tabel ringkasan adalah rata-rata tiga skenario dengan bobot sama. Jumlah data tiap skenario adalah 25, 25, dan 80, sehingga rata-rata ini berbeda dari akurasi gabungan 130 data. Tiap sistem menghasilkan 130 prediksi; dua sistem menghasilkan 260 baris prediksi.

| Perintah | Fungsi |
|---|---|
| `python scripts/evaluate.py` | Jalankan semua prediksi, tulis `results/*.csv` dan `results/evaluation_report.md` |
| `python scripts/evaluate.py --reuse` | Bangun ulang tabel dari `results/predictions.csv` |

Keluaran: `predictions.csv` (satu baris per sistem x data uji, dengan biaya ke 5 vowel), `table0_summary.csv`,
`table1_per_vowel.csv`, `table2_closed_per_speaker.csv`, `table3_open_per_speaker.csv`, confusion matrix dan daftar kesalahan.
