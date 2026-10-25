# Hasil Evaluasi (Person 4)

## Ringkasan akurasi (%)

| Sistem | Closed Recording | Closed Speakers | Open Speakers | Rata-rata |
|---|---|---|---|---|
| DTW Template | 96.0% | 92.0% | 86.2% | 91.4% |
| Model Gaussian | 100.0% | 100.0% | 78.8% | 92.9% |

## Tabel 1: Akurasi per vowel

| Vowel | DTW Closed Recording | DTW Closed Speakers | DTW Open Speakers | Gaussian Closed Recording | Gaussian Closed Speakers | Gaussian Open Speakers |
|---|---|---|---|---|---|---|
| a | 100.0% | 100.0% | 93.8% | 100.0% | 100.0% | 93.8% |
| i | 100.0% | 100.0% | 93.8% | 100.0% | 100.0% | 81.2% |
| u | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 93.8% |
| e | 100.0% | 80.0% | 93.8% | 100.0% | 100.0% | 62.5% |
| o | 80.0% | 80.0% | 50.0% | 100.0% | 100.0% | 62.5% |
| Average Accuracy | 96.0% | 92.0% | 86.2% | 100.0% | 100.0% | 78.8% |

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
| external_g03_s01 | 100.0% | 80.0% | 90.0% |
| external_g03_s02 | 60.0% | 60.0% | 60.0% |
| external_g03_s03 | 60.0% | 80.0% | 70.0% |
| external_g03_s05 | 100.0% | 100.0% | 100.0% |
| external_g04_s03 | 80.0% | 80.0% | 80.0% |
| external_g04_s04 | 60.0% | 60.0% | 60.0% |
| external_g07_s01 | 80.0% | 100.0% | 90.0% |
| external_g07_s02 | 80.0% | 60.0% | 70.0% |
| external_g07_s03 | 100.0% | 100.0% | 100.0% |
| external_g07_s04 | 100.0% | 60.0% | 80.0% |
| external_g07_s05 | 80.0% | 80.0% | 80.0% |
| external_g09_s01 | 100.0% | 60.0% | 80.0% |
| external_g09_s02 | 100.0% | 100.0% | 100.0% |
| external_g09_s03 | 80.0% | 80.0% | 80.0% |
| external_g09_s04 | 100.0% | 80.0% | 90.0% |
| external_g09_s05 | 100.0% | 80.0% | 90.0% |
| Average | 86.2% | 78.8% | 82.5% |

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
| a | 15 | 0 | 0 | 1 | 0 |
| i | 0 | 15 | 1 | 0 | 0 |
| u | 0 | 0 | 16 | 0 | 0 |
| e | 0 | 0 | 0 | 15 | 1 |
| o | 4 | 0 | 4 | 0 | 8 |

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
| a | 15 | 0 | 0 | 0 | 1 |
| i | 0 | 13 | 1 | 2 | 0 |
| u | 0 | 0 | 15 | 0 | 1 |
| e | 0 | 1 | 4 | 10 | 1 |
| o | 2 | 0 | 4 | 0 | 10 |

## Daftar kesalahan

| Sistem | Skenario | Speaker | Benar | Prediksi | File |
|---|---|---|---|---|---|
| DTW Template | Closed Recording | speaker02 | o | u | speaker02_o_01.wav |
| DTW Template | Closed Speakers | speaker02 | e | i | speaker02_e_02.wav |
| DTW Template | Closed Speakers | speaker02 | o | u | speaker02_o_02.wav |
| Model Gaussian | Open Speakers | external_g03_s01 | e | i | external_g03_s01_e_01.wav |
| DTW Template | Open Speakers | external_g03_s02 | a | e | external_g03_s02_a_01.wav |
| Model Gaussian | Open Speakers | external_g03_s02 | a | o | external_g03_s02_a_01.wav |
| DTW Template | Open Speakers | external_g03_s02 | o | u | external_g03_s02_o_01.wav |
| Model Gaussian | Open Speakers | external_g03_s02 | o | u | external_g03_s02_o_01.wav |
| Model Gaussian | Open Speakers | external_g03_s03 | e | u | external_g03_s03_e_01.wav |
| DTW Template | Open Speakers | external_g03_s03 | i | u | external_g03_s03_i_01.wav |
| DTW Template | Open Speakers | external_g03_s03 | o | u | external_g03_s03_o_01.wav |
| Model Gaussian | Open Speakers | external_g04_s03 | i | e | external_g04_s03_i_01.wav |
| DTW Template | Open Speakers | external_g04_s03 | o | a | external_g04_s03_o_01.wav |
| DTW Template | Open Speakers | external_g04_s04 | e | o | external_g04_s04_e_01.wav |
| Model Gaussian | Open Speakers | external_g04_s04 | e | u | external_g04_s04_e_01.wav |
| DTW Template | Open Speakers | external_g04_s04 | o | a | external_g04_s04_o_01.wav |
| Model Gaussian | Open Speakers | external_g04_s04 | o | a | external_g04_s04_o_01.wav |
| DTW Template | Open Speakers | external_g07_s01 | o | a | external_g07_s01_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s02 | e | u | external_g07_s02_e_01.wav |
| DTW Template | Open Speakers | external_g07_s02 | o | a | external_g07_s02_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s02 | o | a | external_g07_s02_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s04 | e | u | external_g07_s04_e_01.wav |
| Model Gaussian | Open Speakers | external_g07_s04 | i | u | external_g07_s04_i_01.wav |
| DTW Template | Open Speakers | external_g07_s05 | o | u | external_g07_s05_o_01.wav |
| Model Gaussian | Open Speakers | external_g07_s05 | o | u | external_g07_s05_o_01.wav |
| Model Gaussian | Open Speakers | external_g09_s01 | e | o | external_g09_s01_e_01.wav |
| Model Gaussian | Open Speakers | external_g09_s01 | i | e | external_g09_s01_i_01.wav |
| DTW Template | Open Speakers | external_g09_s03 | o | u | external_g09_s03_o_01.wav |
| Model Gaussian | Open Speakers | external_g09_s03 | o | u | external_g09_s03_o_01.wav |
| Model Gaussian | Open Speakers | external_g09_s04 | o | u | external_g09_s04_o_01.wav |
| Model Gaussian | Open Speakers | external_g09_s05 | u | o | external_g09_s05_u_01.wav |
