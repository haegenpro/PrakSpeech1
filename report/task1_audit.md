# Audit Tugas 1: Dataset dan MFCC

Fitur sudah benar. Ekstraksi ulang di laptop lain menghasilkan 75 `.npy` identik (selisih 0.0).

| Item | Status |
|---|---|
| MFCC 39 = 13 statis + 13 delta + 13 delta-delta | Benar |
| Template = rekaman 01, closed = rekaman 02, open = G07 | Benar, tidak bocor |
| Path absolut `C:\Users\haege\...` di `metadata.csv` dan `manifest.json` | **Sudah diperbaiki**, sekarang path relatif |
| Hening tidak dipotong (`speaker01_a_01`: 478 frame, vokal ~27 frame) | Ditangani di `src/silence.py` |
| Versi library ikut dicek, beda versi wajib `--overwrite` | Saran: cek konfigurasi saja |
| `dataset_utils.py` 296 baris (SHA256, atomic write, manifest impor) | Saran: sederhanakan |
| README memakai em dash dan terkesan buatan AI | Saran: tulis ulang singkat |
| Rekaman speaker01 ~4.8 s, vokal hanya ~0.3 s | Saran: rekam ulang ~1 s |
