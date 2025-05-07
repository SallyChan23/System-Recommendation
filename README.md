# Laporan Proyek Machine Learning - Jeselyn Tania

## Project Overview

Sistem rekomendasi telah menjadi tulang punggung dari berbagai layanan digital, termasuk dalam industri musik. Dengan banyaknya pilihan lagu yang tersedia secara online, pengguna seringkali kesulitan menemukan musik yang sesuai dengan selera mereka tanpa bantuan sistem pintar. Proyek ini bertujuan untuk membangun sistem rekomendasi lagu berbasis karakteristik audio menggunakan pendekatan Content-Based Filtering.

Masalah ini penting untuk diselesaikan karena preferensi musik sangat subjektif dan dapat dipersonalisasi dengan baik jika sistem memahami ciri khas lagu yang disukai pengguna. Dengan memanfaatkan data fitur audio dari Spotify seperti danceability, energy, valence, dan lainnya, sistem dapat menyarankan lagu-lagu yang secara statistik mirip dengan lagu yang disukai.

Berdasarkan riset oleh Celma (2010), sistem rekomendasi musik berbasis konten dapat membantu pengguna menemukan lagu baru secara eksploratif sekaligus meningkatkan keterlibatan pengguna pada platform streaming musik [1]. Oleh karena itu, proyek ini tidak hanya bermanfaat dari sisi teknis, tetapi juga memberikan nilai bisnis pada peningkatan user engagement.

### Referensi:
[1] Ò. Celma, *Music Recommendation and Discovery: The Long Tail, Long Fail, and Long Play in the Digital Music Space*, Springer, 2010.

## Business Understanding

### Problem Statements

- Pengguna seringkali kesulitan menemukan lagu yang sesuai dengan preferensi mereka di tengah jutaan pilihan lagu yang tersedia di platform streaming.
- Rekomendasi yang muncul pada platform terkadang terlalu umum dan tidak sesuai dengan selera personal pengguna.
- Tidak adanya sistem yang secara otomatis menyarankan lagu berdasarkan kemiripan karakteristik audio dengan lagu yang disukai pengguna.

### Goals

- Membangun sistem rekomendasi lagu yang mampu memberikan saran lagu-lagu lain berdasarkan kemiripan karakteristik audio.
- Memberikan pengalaman personalisasi musik yang lebih relevan kepada pengguna dengan pendekatan content-based filtering.

### Solution Approach

Untuk menjawab permasalahan di atas, dua pendekatan sistem rekomendasi dianalisis dan dipilih:

1. **Content-Based Filtering**  
   Sistem akan menyarankan lagu berdasarkan kemiripan fitur audio (seperti danceability, energy, valence, dll) dengan lagu yang disukai. Pendekatan ini cocok jika data interaksi pengguna (seperti rating atau riwayat dengar) tidak tersedia.

2. **Collaborative Filtering** *(tidak digunakan dalam proyek ini)*  
   Biasanya menggunakan data user-item interaction (misalnya riwayat lagu yang disukai pengguna lain yang mirip). Namun, karena dataset yang digunakan tidak menyertakan data pengguna, pendekatan ini hanya disebut sebagai alternatif.

Dalam proyek ini, pendekatan **Content-Based Filtering** dipilih karena sesuai dengan karakteristik data dan tujuan yang ingin dicapai.

## Data Understanding

Pada bagian ini, akan dijelaskan informasi mengenai data yang digunakan dalam proyek machine learning ini. Data yang digunakan berasal dari sumber terbuka dan berisi informasi mengenai karakteristik audio dari lebih dari 160.000 lagu yang tersedia di platform Spotify. Tujuan utama dari tahap ini adalah untuk memahami struktur, kondisi, dan karakteristik fitur-fitur dalam dataset sebelum dilakukan preprocessing dan pemodelan sistem rekomendasi.

### Sumber Data

Dataset diambil dari Kaggle dengan judul **Spotify Dataset 1921–2020, 160k+ Tracks**, yang dapat diakses melalui tautan berikut:  
🔗 [https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-1921-2020-160k-tracks)

### Variabel-variabel pada dataset ini antara lain:

- `name`: Judul lagu.
- `artists`: Artis atau penyanyi dari lagu.
- `release_date`: Tanggal rilis lagu.
- `year`: Tahun rilis lagu.
- `duration_ms`: Durasi lagu dalam milidetik.
- `explicit`: Indikator apakah lagu mengandung konten eksplisit.
- `popularity`: Skor popularitas dari Spotify.
- `danceability`: Seberapa cocok lagu untuk menari.
- `energy`: Intensitas dan aktivitas lagu.
- `key`: Tangga nada lagu (0–11).
- `loudness`: Tingkat volume rata-rata lagu (dalam dB).
- `mode`: Mayor (1) atau minor (0).
- `speechiness`: Seberapa banyak elemen spoken-word.
- `acousticness`: Tingkat ke-akustikan lagu.
- `instrumentalness`: Kemungkinan lagu bersifat instrumental.
- `liveness`: Indikator apakah lagu direkam secara live.
- `valence`: Positif atau negatifnya mood lagu.
- `tempo`: Kecepatan lagu dalam BPM.

### Jumlah Data

Dataset ini terdiri dari **169.909 baris (observasi)** dan **19 kolom (fitur)**.

### Kondisi Data

- **Missing Value**: Tidak terdapat missing value pada dataset.
- **Data Duplikat**: Tidak ditemukan data duplikat.
- **Outlier**: Outlier tidak ditangani secara eksplisit karena fokus utama adalah pada sistem rekomendasi, dan tidak terdapat distribusi ekstrem yang mengganggu perhitungan kemiripan fitur.

### Exploratory Data Analysis (EDA)

Sebagai bagian dari tahap pemahaman data, dilakukan eksplorasi awal terhadap dataset menggunakan fungsi-fungsi seperti `df.info()`, `df.describe()`, `df.isnull().sum()` dan `df.duplicated().sum()`. Tahapan ini membantu dalam memahami tipe data, distribusi nilai, serta memastikan bahwa data dalam kondisi bersih dan siap untuk diproses lebih lanjut. Visualisasi data seperti heatmap dan PCA akan digunakan pada tahap evaluasi untuk menilai seberapa baik sistem mengelompokkan lagu-lagu berdasarkan kemiripan fitur audionya.

## Data Preparation

Tahap data preparation dilakukan untuk mempersiapkan data agar dapat digunakan oleh sistem rekomendasi dengan optimal. Karena proyek ini menggunakan pendekatan Content-Based Filtering, maka proses difokuskan pada pemilihan fitur yang relevan dan normalisasi data agar perhitungan kemiripan (similarity) dapat dilakukan dengan baik.

Berikut adalah tahapan-tahapan yang dilakukan dalam proses data preparation:

### 1. Pemilihan Fitur Relevan

Langkah pertama yang dilakukan adalah memilih fitur-fitur audio yang akan digunakan sebagai dasar perhitungan kemiripan antar lagu. Fitur yang dipilih merupakan hasil ekstraksi dari Spotify API dan secara langsung merepresentasikan karakteristik musik, di antaranya:

- `danceability`
- `energy`
- `key`
- `loudness`
- `mode`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`

Fitur-fitur ini dipilih karena mampu mencerminkan ritme, mood, struktur, dan intensitas dari sebuah lagu.

### 2. Normalisasi Fitur (Feature Scaling)

Karena masing-masing fitur memiliki skala nilai yang berbeda (contoh: tempo dalam BPM bisa bernilai ratusan, sedangkan acousticness bernilai antara 0–1), maka dilakukan normalisasi menggunakan `MinMaxScaler` dari Scikit-Learn. Normalisasi ini mengubah semua nilai fitur ke rentang antara 0 hingga 1.

Alasan dilakukan normalisasi:
- Untuk menghindari bias pada fitur yang memiliki skala besar
- Agar semua fitur diperlakukan secara setara dalam proses perhitungan cosine similarity

### 3. Penambahan Informasi Referensi

Setelah fitur audio dinormalisasi, informasi pendukung seperti `track_name` dan `artist_name` tetap dipertahankan di dataframe akhir agar hasil rekomendasi dapat ditampilkan dengan jelas (nama lagu dan artis). Kedua kolom ini tidak digunakan dalam proses perhitungan kemiripan, tetapi berfungsi sebagai identitas pada output sistem.

### 4. Sampling Data

Dataset asli berisi lebih dari 160.000 lagu. Untuk menghindari overload memori pada saat proses perhitungan cosine similarity (karena matrix 160k x 160k akan sangat besar), dilakukan pengambilan sampel sebanyak 10.000 lagu secara acak. Jumlah ini sudah mencukupi untuk demonstrasi dan evaluasi sistem rekomendasi secara efisien tanpa membuat sistem crash.

Sampling dilakukan menggunakan `df.sample(n=10000, random_state=42)` untuk menjaga reprodusibilitas hasil.

## Modeling

Dalam proyek ini, sistem rekomendasi dibangun menggunakan pendekatan **Content-Based Filtering** dengan perhitungan **Cosine Similarity** antar lagu berdasarkan fitur audionya. Meskipun tidak menggunakan model supervised seperti regresi atau klasifikasi, proses modeling tetap melibatkan pemilihan algoritma, transformasi data, dan perhitungan kemiripan yang menghasilkan rekomendasi.

Model: Content-Based Filtering dengan Cosine Similarity

### Cara Kerja:

Content-Based Filtering merekomendasikan item berdasarkan kemiripan antara item yang satu dengan lainnya. Pada proyek ini, setiap lagu direpresentasikan sebagai vektor numerik berdasarkan 11 fitur audio (seperti danceability, energy, valence, tempo, dll).  
Kemudian, digunakan **Cosine Similarity** untuk mengukur kemiripan antar lagu. Nilai cosine similarity berkisar antara 0 hingga 1, di mana nilai yang mendekati 1 menunjukkan bahwa dua lagu memiliki karakteristik yang sangat mirip.

Rumus Cosine Similarity:  
$similarity = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \times \|\vec{B}\|}$

### Parameter:

- Jumlah sampel: 10.000 lagu (diambil secara acak dari total 169.909 lagu)
- Fitur: 11 fitur audio numerik
- Algoritma kemiripan: `cosine_similarity` dari `sklearn.metrics.pairwise`

### Implementasi:

1. Setiap lagu direpresentasikan sebagai vektor dari fitur audio yang telah dinormalisasi.
2. Dibuat similarity matrix berdimensi 10.000 x 10.000 yang menunjukkan skor kemiripan antar lagu.
3. Dibuat fungsi `rekomendasi_lagu()` yang akan mengeluarkan daftar lagu yang paling mirip dengan input berdasarkan similarity score tertinggi.

---

### Kelebihan dan Kekurangan Model

**Content-Based Filtering (Cosine Similarity)**

- **Kelebihan:**
  - Tidak membutuhkan data pengguna (user-item interaction)
  - Dapat menangkap kemiripan antar item (lagu) berdasarkan konten deskriptif
  - Hasil rekomendasi cukup personal jika pengguna memberikan lagu acuan yang tepat

- **Kekurangan:**
  - Rekomendasi terbatas pada kemiripan konten saja (tidak bisa menangkap tren kolektif seperti collaborative filtering)
  - Tidak ada "surprise discovery" karena cenderung merekomendasikan item yang mirip dengan yang sudah diketahui
  - Perhitungan similarity matrix bisa berat secara komputasi pada dataset besar (oleh karena itu dilakukan sampling 10k lagu)

---

### Model Selection

Karena proyek ini tidak memiliki informasi pengguna dan bertujuan merekomendasikan lagu berdasarkan karakteristik musik itu sendiri, maka **Content-Based Filtering dengan Cosine Similarity** merupakan pendekatan yang paling sesuai dan efisien. Selain itu, hasil evaluasi sistem menunjukkan bahwa rekomendasi yang dihasilkan secara umum masuk akal dan relevan secara musikal.

Model ini juga mudah untuk diinterpretasikan dan dapat dikembangkan lebih lanjut dengan tambahan metadata seperti genre, mood, atau lirik.

## Evaluation

Evaluasi pada sistem rekomendasi ini dilakukan dengan pendekatan berbasis studi kasus. Karena menggunakan algoritma unsupervised (tanpa label ground-truth), evaluasi dilakukan secara **kualitatif**, yaitu dengan melihat apakah rekomendasi lagu yang diberikan relevan dan masuk akal secara musikal.

### 1. Evaluasi Fungsi Rekomendasi

Dibuat fungsi `rekomendasi_lagu(judul_lagu)` yang akan mengembalikan daftar lagu paling mirip berdasarkan perhitungan cosine similarity dari fitur audio.

Contoh output:

![image](https://github.com/user-attachments/assets/60472720-1d22-4d3d-b848-e5bf90415728)

> Lagu-lagu yang direkomendasikan memiliki skor similarity tinggi (0.98–0.99), menunjukkan bahwa karakteristik audio-nya sangat mirip.

### 2. Penanganan Kasus Lagu Tidak Ditemukan

Sistem juga dirancang untuk dapat menangani input lagu yang tidak terdapat dalam dataset. Jika lagu tidak ditemukan, maka sistem akan memberikan pesan berikut:

![image](https://github.com/user-attachments/assets/6d9a2ae7-cd3c-4772-bbaf-04b06903a38e)

> Fitur ini memastikan bahwa sistem tetap robust dan user-friendly terhadap input yang tidak valid.

### 3. Visualisasi Evaluasi

Sebagai bagian dari evaluasi, dilakukan dua jenis visualisasi:

- **Heatmap Cosine Similarity**
  ![image](https://github.com/user-attachments/assets/47b268d2-6227-47e9-a9e7-4fdc2fe532d7)

  > Untuk melihat sejauh mana kemiripan antar lagu dalam subset data. Warna biru tua menunjukkan nilai similarity tinggi antar lagu.

- **Visualisasi PCA (Principal Component Analysis)**
  ![image](https://github.com/user-attachments/assets/41671fb8-3974-459a-96b1-900e0d5118b4)

  > Digunakan untuk memproyeksikan seluruh dataset lagu ke dalam 2 dimensi berdasarkan fitur audio-nya. Hasil menunjukkan bahwa lagu dengan fitur mirip cenderung mengelompok bersama.

Visualisasi ini memperkuat keyakinan bahwa pendekatan cosine similarity berhasil memetakan lagu berdasarkan karakteristik audionya dengan cukup baik.

### Evaluation Metric Explanation

Karena sistem ini menggunakan pendekatan **Content-Based Filtering dengan cosine similarity**, maka metrik evaluasi utamanya adalah:

#### Cosine Similarity

Cosine Similarity mengukur tingkat kemiripan antara dua vektor (dalam hal ini: fitur audio lagu) berdasarkan sudut antar vektor. Semakin kecil sudutnya (semakin dekat nilainya ke 1), maka semakin mirip dua lagu tersebut.

**Rumus Cosine Similarity:**
$similarity = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \times \|\vec{B}\|}$

## Kesimpulan

- Sistem rekomendasi berhasil dibangun menggunakan pendekatan Content-Based Filtering.
- Model menggunakan Cosine Similarity untuk mengukur kemiripan antar lagu berdasarkan 11 fitur audio.
- Hasil rekomendasi menunjukkan kemiripan lagu yang logis dan relevan secara musikal.
- Visualisasi PCA dan heatmap membantu menunjukkan pola dan pengelompokan lagu yang serupa.
- Fungsi evaluasi menunjukkan sistem mampu menangani input valid dan invalid dengan baik.

## Referensi

- [Spotify Dataset 1921–2020 - 160k Tracks (Kaggle)](https://www.kaggle.com/datasets/mrmorj/spotify-dataset-19212020-160k-tracks)
- Dokumentasi Scikit-Learn
- Dokumentasi Seaborn & Matplotlib
