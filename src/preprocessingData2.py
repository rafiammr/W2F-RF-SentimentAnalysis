import re
import pandas as pd
import nltk
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Pastikan NLTK sudah memiliki dataset yang dibutuhkan
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
file_path = "NewDatasetLivin.csv"
df = pd.read_csv(file_path)

slang_df = pd.read_csv("normalization_dict.csv")
slang_dict = dict(zip(slang_df["slang"], slang_df["formal"]))

negations_df = pd.read_csv("negations.csv")
negations = set(negations_df["negation"])

antonim_df = pd.read_csv("antonim_dict.csv")
antonim_dict = dict(zip(antonim_df["word"], antonim_df["antonym"]))

stopwords_df = pd.read_csv("custom_stopwords.csv")
custom_stopwords = set(stopwords_df["stopword"])

keep_words_df = pd.read_csv("keep_words.csv")
keep_words = set(keep_words_df["keep_word"])


def normalize_word(word):
    return slang_dict.get(word, word)


factory = StemmerFactory()
stemmer = factory.create_stemmer()


def handle_negation(tokens):
    new_tokens = []
    negate = False

    for word in tokens:
        if word in negations:
            new_tokens.append("tidak_")
            negate = True
        elif negate:
            antonim_word = antonim_dict.get(word, None)
            if antonim_word:
                new_tokens[-1] = antonim_word
            else:
                new_tokens[-1] = new_tokens[-1] + word
            negate = False
        else:
            new_tokens.append(word)

    return new_tokens


stop_words = set(stopwords.words("indonesian")) - custom_stopwords


def preprocess_text(text):
    if not isinstance(text, str):
        return []

    cleansing = re.sub(r"[^a-zA-Z\s]", " ", text)

    case_folding = cleansing.lower()

    tokens = case_folding.split()

    normalized = [normalize_word(word) for word in tokens]

    stemming = [
        stemmer.stem(word) if word not in keep_words else word for word in normalized
    ]

    negation = handle_negation(stemming)

    stopword = [word for word in negation if word not in stop_words]

    return stopword


# Preprocessing semua ulasan dengan tqdm untuk progress bar
tqdm.pandas(desc="Preprocessing ulasan")
df["text_cleaned"] = df["content"].progress_apply(preprocess_text)

# Simpan ke CSV: ubah list menjadi string JSON
df["text_cleaned"] = df["text_cleaned"].apply(json.dumps)

# Simpan hasil preprocessing ke file baru
processed_file_path = "DatasetLivinPreproNewFinal.csv"
df.to_csv(processed_file_path, index=False)

print(
    "Preprocessing selesai! Hasil disimpan di 'livin_by_mandiri_100k_preprocessed.csv'"
)
