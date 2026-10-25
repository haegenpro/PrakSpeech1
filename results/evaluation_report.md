# Hasil Evaluasi (Person 4)

## Ringkasan akurasi (%)

| Sistem | Closed Recording | Closed Speakers | Open Speakers | Rata-rata |
|---|---|---|---|---|
| DTW Template | 96.0% | 92.0% | 88.0% | 92.0% |
| Model Gaussian | 100.0% | 100.0% | 80.0% | 93.3% |

## Tabel 1: Akurasi per vowel

| Vowel | DTW Closed Recording | DTW Closed Speakers | DTW Open Speakers | Gaussian Closed Recording | Gaussian Closed Speakers | Gaussian Open Speakers |
|---|---|---|---|---|---|---|
| a | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| i | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 80.0% |
| u | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| e | 100.0% | 80.0% | 100.0% | 100.0% | 100.0% | 60.0% |
| o | 80.0% | 80.0% | 40.0% | 100.0% | 100.0% | 60.0% |
| Average Accuracy | 96.0% | 92.0% | 88.0% | 100.0% | 100.0% | 80.0% |

## Tabel 2: Closed experiments per speaker

| Speaker | DTW Closed Recording | Gaussian Closed Recording | DTW Closed Speakers | Gaussian Closed Speakers | Average |
|---|---|---|---|---|---|
| speaker01 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speaker02 | 80.0% | 100.0% | 60.0% | 100.0% | 85.0% |
| speaker03 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speaker04 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speaker05 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Average | 96.0% | 100.0% | 92.0% | 100.0% | 97.0% |

## Tabel 3: Open experiments per speaker

| Speaker (open) | DTW Open Speakers | Gaussian Open Speakers | Average |
|---|---|---|---|
| external_g07_s01 | 80.0% | 100.0% | 90.0% |
| external_g07_s02 | 80.0% | 60.0% | 70.0% |
| external_g07_s03 | 100.0% | 100.0% | 100.0% |
| external_g07_s04 | 100.0% | 60.0% | 80.0% |
| external_g07_s05 | 80.0% | 80.0% | 80.0% |
| Average | 88.0% | 80.0% | 84.0% |

## Confusion matrix

### DTW Template - Closed Recording

| benar \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 5 | 0 | 0 | 0 | 0 |
| i | 0 | 5 | 0 | 0 | 0 |
| u | 0 | 0 | 5 | 0 | 0 |
| e | 0 | 0 | 0 | 5 | 0 |
| o | 0 | 0 | 1 | 0 | 4 |

### DTW Template - Closed Speakers

| benar \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 5 | 0 | 0 | 0 | 0 |
| i | 0 | 5 | 0 | 0 | 0 |
| u | 0 | 0 | 5 | 0 | 0 |
| e | 0 | 1 | 0 | 4 | 0 |
| o | 0 | 0 | 1 | 0 | 4 |

### DTW Template - Open Speakers

| benar \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 5 | 0 | 0 | 0 | 0 |
| i | 0 | 5 | 0 | 0 | 0 |
| u | 0 | 0 | 5 | 0 | 0 |
| e | 0 | 0 | 0 | 5 | 0 |
| o | 2 | 0 | 1 | 0 | 2 |

### Model Gaussian - Closed Recording

| benar \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 5 | 0 | 0 | 0 | 0 |
| i | 0 | 5 | 0 | 0 | 0 |
| u | 0 | 0 | 5 | 0 | 0 |
| e | 0 | 0 | 0 | 5 | 0 |
| o | 0 | 0 | 0 | 0 | 5 |

### Model Gaussian - Closed Speakers

| benar \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 5 | 0 | 0 | 0 | 0 |
| i | 0 | 5 | 0 | 0 | 0 |
| u | 0 | 0 | 5 | 0 | 0 |
| e | 0 | 0 | 0 | 5 | 0 |
| o | 0 | 0 | 0 | 0 | 5 |

### Model Gaussian - Open Speakers

| benar \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 5 | 0 | 0 | 0 | 0 |
| i | 0 | 4 | 1 | 0 | 0 |
| u | 0 | 0 | 5 | 0 | 0 |
| e | 0 | 0 | 2 | 3 | 0 |
| o | 1 | 0 | 1 | 0 | 3 |

## Daftar kesalahan

| Sistem | Skenario | Speaker | Benar | Prediksi | File |
|---|---|---|---|---|---|
| DTW Template | Closed Recording | speaker02 | o | u | speaker02_o_01.wav |
| DTW Template | Closed Speakers | speaker02 | e | i | speaker02_e_02.wav |
| DTW Template | Closed Speakers | speaker02 | o | u | speaker02_o_02.wav |
| DTW Template | Open Speakers | external_g07_s01 | o | a | external_g07_s01_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s02 | e | u | external_g07_s02_e_01.wav |
| DTW Template | Open Speakers | external_g07_s02 | o | a | external_g07_s02_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s02 | o | a | external_g07_s02_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s04 | e | u | external_g07_s04_e_01.wav |
| Model Gaussian | Open Speakers | external_g07_s04 | i | u | external_g07_s04_i_01.wav |
| DTW Template | Open Speakers | external_g07_s05 | o | u | external_g07_s05_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s05 | o | u | external_g07_s05_o_01.wav |
