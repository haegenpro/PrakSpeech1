# Audit Tugas 1: Dataset dan MFCC

Audit awal mencakup 75 `.npy` yang identik pada ekstraksi di laptop lain. Setelah penambahan data, dataset berisi 130 rekaman: 25 template, 25 closed-speaker, dan 80 open-speaker dari 16 orang. Seluruh hash audio unik sebelum dan sesudah migrasi identik; salinan sumber yang terverifikasi telah dihapus.

| Item | Status |
|---|---|
| MFCC 39 = 13 statis + 13 delta + 13 delta-delta | Benar |
| Template = rekaman 01, closed = rekaman 02, open = G03/G04/G07/G09 | Benar, tidak bocor |
| Path absolut `C:\Users\haege\...` di `metadata.csv` dan `manifest.json` | **Sudah diperbaiki**, sekarang path relatif |
| Hening tidak dipotong (`speaker01_a_01`: 478 frame, vokal ~27 frame) | Ditangani di `src/silence.py` |
| Versi library ikut dicek, beda versi wajib `--overwrite` | Saran: cek konfigurasi saja |
| `dataset_utils.py` (SHA256, atomic write, manifest impor) | Saran: sederhanakan |
| README memakai em dash dan terkesan buatan AI | Saran: tulis ulang singkat |
| Rekaman speaker01 ~4.8 s, vokal hanya ~0.3 s | Saran: rekam ulang ~1 s |
