# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('/kaggle/input/spotify-dataset-1921-2020-160k-tracks/data.csv')

# EDA
df.info()
df.describe()
print("Jumlah missing value:")
print(df.isnull().sum())
print("Jumlah data duplikat:", df.duplicated().sum())

# Pemilihan fitur audio
fitur_audio = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

df[fitur_audio].head()
print(df.columns.tolist())

# Data Preparation
scaler = MinMaxScaler()
audio_scaled = scaler.fit_transform(df[fitur_audio])
df_scaled = pd.DataFrame(audio_scaled, columns=fitur_audio)
df_scaled['track_name'] = df['name']
df_scaled['artist_name'] = df['artists']

# Modeling
df_sample = df_scaled.sample(n=10000, random_state=42)
similarity_matrix = cosine_similarity(df_sample[fitur_audio])
similarity_df = pd.DataFrame(similarity_matrix,
                             index=df_sample['track_name'],
                             columns=df_sample['track_name'])

# Fungsi Rekomendasi
def rekomendasi_lagu(judul_lagu, df_similar, jumlah=5):
    if judul_lagu not in df_similar.columns:
        return f"Lagu '{judul_lagu}' tidak ditemukan dalam dataset."
    
    similar_songs = df_similar[judul_lagu].sort_values(ascending=False).iloc[1:jumlah+1]
    hasil_df = pd.DataFrame({
        'Track Name': similar_songs.index,
        'Similarity Score': similar_songs.values
    })
    return hasil_df

def evaluasi_lagu(list_judul, df_similar):
    for judul in list_judul:
        print(f"\nJudul: {judul}")
        hasil = rekomendasi_lagu(judul, df_similar)
        if isinstance(hasil, str):
            print("❌", hasil)
        else:
            print("✅ Rekomendasi lagu mirip:")
            print(hasil)

# Evaluasi
lagu_uji = [
    'Camby Bolongo', 'Castle on a Cloud', 'Someone Like You',
    'Eso Hey Sajal Shyam Ghana Deya', 'Bohemian Rhapsody'
]
evaluasi_lagu(lagu_uji, similarity_df)

# Visualisasi Heatmap
subset = similarity_df.iloc[:10, :10]
plt.figure(figsize=(10, 8))
sns.heatmap(subset, annot=False, cmap="YlGnBu")
plt.title('Cosine Similarity antar Lagu (Subset)')
plt.show()

# Visualisasi PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_sample[fitur_audio])
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title('Distribusi Lagu Berdasarkan Fitur Audio (PCA)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
