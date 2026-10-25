# Metodologi Tugas 1: Persiapan Dataset dan Ekstraksi MFCC

## A. Persiapan Dataset

Kamus pengenalan terdiri atas lima kelas vokal Bahasa Indonesia, yaitu /a/, /i/,
/u/, /e/, dan /o/. Dataset internal dirancang menggunakan lima pembicara. Setiap
pembicara menyediakan dua rekaman berbeda untuk setiap vokal, sehingga kebutuhan
dataset internal adalah 5 × 5 × 2 = 50 berkas WAV. Perekaman dianjurkan dilakukan
pada lingkungan tenang dengan jarak mikrofon dan penguatan yang konsisten.
Setiap berkas memuat satu pengucapan vokal; pengulangan berupa pengucapan baru,
bukan salinan rekaman pertama.

Identitas internal ditetapkan sebagai speaker01 (Peter Wongsoredjo), speaker02
(Jessica Allen), speaker03 (Haegen Quinston), speaker04 (Nathaniel JR), dan
speaker05 (Maheswara Bayu Kaindra). Variasi nama pada berkas sumber dipetakan
secara eksplisit, termasuk Peter/S01, Nathan/Nathaniel, serta Indra. Setiap rekaman
disimpan dengan nama baku pada direktori kanonis. Skrip setup memeriksa seluruh
pemetaan dan membandingkan isi berkas secara langsung sebelum menyalin. Tujuan
yang identik dilewati, konflik ditolak, dan sumber dipertahankan. Dataset yang
sudah kanonis tidak memerlukan impor ulang atau manifest impor.

## B. Pembagian Data

Rekaman pertama setiap pasangan pembicara–vokal digunakan pada partisi template,
sedangkan rekaman kedua digunakan pada partisi pengujian closed-speaker. Setiap
partisi internal diharapkan berisi 25 rekaman. Pembagian ini memisahkan pengucapan
untuk pelatihan dan pengujian, meskipun pembicaranya sama. Skenario closed-recording
dapat menggunakan kembali fitur template tanpa membuat partisi baru.

Data open-speaker berasal dari 16 orang: G03/Kel3 (S01, S02, S03, S05), G04
(S03, S04), G07 (S01–S05), dan G09 (S01–S05). Setiap orang menyediakan lima
vokal, sehingga terdapat 80 rekaman eksternal. Identitas seperti
external_g03_s01 mempertahankan kelompok dan nomor pembicara. Direktori audio
adalah template/<speaker>, test/closed_speakers/<speaker>, dan
test/open_speakers/<external_id>. Data eksternal tidak digunakan untuk penyusunan
template maupun estimasi model Gaussian.

## C. Preprocessing Audio

Berkas WAV dibaca menggunakan SoundFile dalam representasi float32. Kanal
dirata-ratakan apabila masukan memiliki lebih dari satu kanal. Berkas kosong,
rusak, bernilai nonfinite, atau menghasilkan sinyal seluruhnya nol ditolak dengan
pesan diagnostik. Frekuensi pencuplikan diseragamkan menjadi 16.000 Hz melalui
resampling polifase menggunakan SciPy apabila frekuensi awal berbeda. Proses
tersebut mengubah sampel sinyal, bukan sekadar metadata frekuensi.

Pemotongan keheningan tepi dan normalisasi amplitudo bersifat opsional serta
dinonaktifkan pada konfigurasi awal. Pemotongan menggunakan ambang relatif terhadap
amplitudo puncak, sedangkan normalisasi membagi sinyal dengan amplitudo puncaknya.
Keduanya dapat memengaruhi transisi dan energi vokal sehingga harus diterapkan
secara konsisten pada seluruh partisi. Tidak digunakan parameter preprocessing
yang dipelajari dari data pengujian.

Operasi tersebut berbeda dari pemotongan frame fitur pada modul DTW dan Gaussian.
Kedua modul secara default mempertahankan rentang frame dengan energi log dalam
20 dB dari puncak. Pemotongan ini dilakukan setelah MFCC dihitung dan tidak
menghitung ulang delta pada batas baru. Kebijakan yang sama diterapkan pada data
latih dan masukan pengenalan. Konfigurasi ekstraksi dipusatkan pada DEFAULT_CONFIG;
CLI hanya menerima direktori data dan direktori output.

## D. Ekstraksi MFCC 39 Dimensi

Praktikum mensyaratkan 39 dimensi per frame. Implementasi ini memilih susunan
13 koefisien MFCC statis, 13 delta, dan 13 delta-delta. Susunan tersebut serta
nilai parameter berikut merupakan keputusan implementasi, bukan ketentuan
numerik tambahan dari petunjuk praktikum.

Ekstraksi menggunakan python_speech_features. Pre-emphasis dengan koefisien 0,97
diterapkan satu kali di dalam fungsi MFCC. Sinyal dibagi menjadi frame sepanjang
25 ms dengan pergeseran 10 ms dan jendela Hamming. Spektrum daya FFT 512 titik
diproyeksikan melalui 26 filter Mel. Logaritma energi filter ditransformasikan
menggunakan DCT untuk memperoleh 13 koefisien. Digunakan cepstral lifter 22 dan
penggantian koefisien pertama dengan log energi frame.

Delta dihitung melalui regresi temporal dengan radius N = 2 frame:

```text
Δc_t = Σ[n(c_(t+n) − c_(t−n))] / [2 Σ n²], untuk n = 1, …, N.
```

Delta-delta diperoleh dengan menerapkan operasi yang sama pada delta. Frame batas
direplikasi untuk perhitungan turunan. Sinyal yang lebih pendek daripada satu
frame diberi padding nol. Hasil berupa matriks float32 berukuran T × 39 dengan T
bergantung pada durasi rekaman. Urutan temporal dipertahankan dan tidak diratakan
menjadi satu vektor per rekaman.

## E. Struktur Output

Setiap rekaman menghasilkan satu berkas NumPy pada hierarki partisi dan pembicara
yang bersesuaian. Sebagai contoh, speaker01_a_01.wav menghasilkan
features/template/speaker01/speaker01_a_01.npy. Metadata CSV menyimpan identitas,
label, lokasi sumber, properti audio sebelum dan sesudah preprocessing, jumlah
frame, dimensi fitur, lokasi fitur, dan status validasi. Path sumber dan audio
berupa path kanonis relatif. Setiap eksekusi menghitung ulang seluruh fitur serta
menulis ulang metadata; tidak ada cache atau manifest ekstraksi. Versi dependensi
dicatat dalam dokumentasi dan requirements sebagai informasi eksperimen, tanpa
menghalangi ekstraksi atau pemuatan ketika versi berubah.

API load_dataset mengembalikan daftar rekaman dengan atribut features, vowel,
speaker, dan file_path. Pemuatan memvalidasi array, identitas, partisi, dan path
relatif tanpa membaca WAV asli. file_path menjadi identitas WAV yang diresolusikan
terhadap direktori data di samping direktori fitur. Loader memberikan urutan frame
utuh; modul DTW dan Gaussian menerapkan pemotongan frame masing-masing. Gaussian
dapat menggabungkan frame menurut vokal dari partisi template untuk estimasi
parameternya sendiri. Tugas 1 tidak menghitung template rata-rata, parameter
Gaussian, prediksi, atau akurasi pengenalan.

## F. Validasi

Validasi memeriksa kelengkapan 50 rekaman internal, kesesuaian nama dan direktori,
pemisahan identitas internal dan eksternal, keberhasilan decoding, serta duplikasi
audio secara eksak, serta kelengkapan lima vokal setiap pembicara eksternal yang
ditemukan. Jumlah pembicara eksternal tidak ditetapkan dalam kode. Setiap matriks
fitur harus berdimensi dua, memiliki tepat
39 kolom, memuat sedikitnya satu frame, dan seluruh nilainya harus finite.
Penyimpanan serta pemuatan diuji untuk menjaga bentuk dan nilai fitur. Pemeriksaan
duplikasi audio hanya dilakukan dalam validasi eksplisit. Hash audio dan fitur
tidak diperiksa saat ekstraksi atau loading; jalankan ekstraksi ulang setelah audio
atau konfigurasi berubah. Berkas fitur hilang atau tidak valid menghasilkan error.

Pengujian perangkat lunak menggunakan sinyal sintetis hanya pada direktori
sementara pytest. Sinyal tersebut tidak dimasukkan ke dataset atau digunakan
sebagai hasil eksperimen pengenalan. Skenario meliputi konversi stereo, resampling,
turunan temporal, masukan sangat pendek, kegagalan decoding, indeks dataset,
konflik impor, persistensi, regenerasi fitur, beberapa kelompok eksternal,
kelengkapan vokal, serta loading tanpa WAV asli atau manifest.

Pada pemeriksaan 25 Oktober 2026 (WIB), ditemukan 25 rekaman template, 25 rekaman
closed-speaker, dan 80 rekaman open-speaker dari 16 pembicara eksternal. Seluruh
130 berkas berhasil didekode dan berformat mono PCM16 pada 16.000 Hz. Tidak
ditemukan rekaman internal hilang, vokal eksternal hilang, nama kanonis tidak valid,
atau duplikasi audio eksak. Ekstraksi menghasilkan 130 matriks tanpa kegagalan,
dengan rentang 33–495 frame dan total 22,147 frame. Pengulangan ekstraksi
menghitung ulang seluruh 130 hasil. Verifikasi refaktor membandingkan array sebelum
dan sesudah regenerasi pada lingkungan yang sama: seluruh bentuk dan nilai identik.
Karena fitur tidak berubah, refaktor ini tidak memerlukan pembangunan ulang model.
Evaluasi yang memakai dataset open lama perlu diulang pada seluruh 80 rekaman.

Angka tersebut merupakan pemeriksaan teknis, bukan bukti ketepatan pelafalan
atau kinerja pengenalan. Perbarui angka apabila data atau konfigurasi berubah.
Jalankan skrip validasi dengan --report untuk menghasilkan ringkasan JSON dan
skrip ekstraksi untuk memperbarui metadata. Akurasi DTW dan Gaussian belum
dihitung karena berada di luar tanggung jawab Tugas 1.
