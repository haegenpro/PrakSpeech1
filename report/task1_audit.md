# Audit Tugas 1: Dataset dan MFCC

Tugas 1 disederhanakan untuk kebutuhan praktikum. Spesifikasi mewajibkan Python,
fitur 39 dimensi per frame, lima vokal, dua rekaman per anggota, serta partisi
template, closed-speaker, dan open-speaker. Parameter MFCC lainnya merupakan
keputusan implementasi.

| Item | Kondisi setelah penyederhanaan |
|---|---|
| MFCC | Tetap 13 statis + 13 delta + 13 delta-delta |
| Internal | 25 template + 25 closed-speaker dari 5 anggota |
| Eksternal | 80 rekaman dari 16 pembicara G03/Kel3, G04, G07, dan G09 |
| Identitas eksternal | Mempertahankan kelompok dan nomor pembicara; ditemukan dinamis |
| Ekstraksi | Selalu menghitung ulang dan menulis metadata CSV |
| Cache dan provenance | Manifest, hash audio/fitur saat ekstraksi/loading, dan pemeriksaan versi dihapus |
| CLI ekstraksi | Hanya --data-dir dan --output-dir; pengaturan bersama di DEFAULT_CONFIG |
| Loading | Memvalidasi array dan metadata tanpa memerlukan WAV asli |
| Impor | Membandingkan byte untuk konflik sebelum menyalin; sumber dipertahankan |
| Validasi eksplisit | Kelengkapan, decoding, duplikasi eksak, dan lima vokal per pembicara eksternal |

Pemotongan waveform sebelum MFCC bersifat opsional dan mati pada konfigurasi awal.
DTW dan Gaussian secara default memotong frame tepi berdasarkan energi log
dengan ambang 20 dB setelah MFCC dihitung. Kedua operasi ini berbeda; pemotongan
frame tidak menghitung ulang delta di batas baru. Banyaknya frame yang dibuang
bukan bukti pasti batas pengucapan vokal.

Verifikasi refaktor membandingkan 130 array sebelum dan sesudah ekstraksi ulang:
bentuk dan seluruh nilai identik pada lingkungan yang sama. Ini mendukung bahwa
refaktor tidak mengubah fitur, bukan membuktikan kualitas pelafalan atau akurasi.
Jumlah frame 22.147 dengan rentang 33–495 per rekaman. Metadata dan laporan validasi
diperbarui; hasil pengenalan tidak diubah oleh refaktor ini.

Versi dependensi dicatat sebagai informasi eksperimen, bukan syarat kompatibilitas.
Ekstraksi perlu diulang setelah audio atau pengaturan berubah. Model tidak perlu
dibangun ulang hanya karena refaktor dengan fitur identik. Evaluasi yang masih
memakai dataset open lama perlu diulang terhadap seluruh 80 rekaman eksternal.
