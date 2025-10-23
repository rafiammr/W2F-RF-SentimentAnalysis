import pandas as pd
from tqdm import tqdm


# Fungsi untuk memberikan label sentimen berdasarkan rating
def label_sentiment(score):
    if score <= 3:
        return "negatif"
    # elif score == 5:
    #    return "positif"
    else:
        return "positif"


# Membaca data CSV yang sudah diproses sebelumnya
file_path = "Newlivin_by_mandiri_60k_prepro.csv"
df = pd.read_csv(file_path)

# Menambahkan label sentimen berdasarkan rating dengan tqdm
if "score" in df.columns:
    print("Melabeli sentimen...")
    df["sentiment"] = [
        label_sentiment(score) for score in tqdm(df["score"], desc="Labeling Sentimen")
    ]
else:
    raise ValueError(
        "Kolom 'rating' tidak ditemukan dalam dataset, pastikan dataset memiliki kolom ini."
    )

# Menyimpan hasil labeling ke file baru
df.to_csv("Newlivin_by_mandiri_60k_preproLabeling.csv", index=False)
print("Labeling selesai! Hasil disimpan di 'ulasan_livin.csv'")
