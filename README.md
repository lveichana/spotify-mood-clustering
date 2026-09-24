# Spotify Mood Clustering

Mengelompokkan lagu berdasarkan karakteristik audio (danceability, energy, acousticness, valence, loudness, instrumentalness) menggunakan K-Means clustering, lalu ditampilkan lewat aplikasi Streamlit interaktif.

## Struktur Project

```
spotify-mood-clustering/
├── README.md
├── data/
│   └── dataset.csv              # dataset mentah (Spotify Tracks Dataset, Kaggle)
├── notebook/
│   └── clustering_spotify.ipynb # notebook analisis lengkap (EDA sampai modeling)
├── model/
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   ├── feature_columns.json
│   ├── cluster_profile.csv
│   └── cluster_songs.csv
├── app.py
└── requirements.txt
```

## Cara Menjalankan Aplikasi

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```
3. Browser akan otomatis terbuka di `http://localhost:8501`.

4. Atau akses aplikasi secara langsung melalui:
**https://spotify-mood-finder.streamlit.app/**

## Cara Menjalankan Notebook

Buka `notebook/clustering_spotify.ipynb` di Jupyter, pastikan `dataset.csv` ada di folder `data/` (atau sesuaikan path `pd.read_csv` di dalam notebook), lalu jalankan seluruh cell secara berurutan (Restart Kernel & Run All disarankan agar hasilnya konsisten).

## Ringkasan Hasil

Dataset Spotify Tracks (114.000 baris menjadi 89.740 lagu unik setelah dedup) dikelompokkan menjadi 5 cluster berdasarkan 6 fitur audio inti. Evaluasi menunjukkan Silhouette Score sekitar 0.266, Davies-Bouldin Index sekitar 1.172, dan Calinski-Harabasz Index sekitar 34.366,5. Kelima cluster diberi nama: Upbeat/Danceable/Happy, Instrumental/Ambient/Calm, Acoustic/Mellow, Instrumental/Electronic (Energic), dan High-Energy/Intense.
