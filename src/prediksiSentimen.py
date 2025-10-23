import joblib
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder

# NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Load kamus slangword
slang_file_path = "colloquial-indonesian-lexicon.csv"
slang_df = pd.read_csv(slang_file_path)
slang_dict = dict(zip(slang_df["slang"], slang_df["formal"]))


# Fungsi untuk mengonversi kalimat ke vektor menggunakan model Word2Vec
def get_sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# Kamus normalisasi
normalization_dict = {
    "perbankan": "bank",
    "ngga": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "nggak": "tidak",
    "enggak": "tidak",
    "gak": "tidak",
    "ngebug": "bug",
    "ngak": "tidak",
    "nggk": "tidak",
    "tdk": "tidak",
    "ngelag": "lag",
    "ngirim": "mengirim",
    "ngasih": "memberi",
    "transaksiin": "transaksi",
    "saldoin": "saldo",
    "terbaik": "baik",
    "rekeningin": "rekening",
    "loginin": "login",
    "mutasiin": "mutasi",
    "topupin": "top-up",
    "memuaskan": "puas",
    "transferin": "transfer",
    "apk": "aplikasi",
    "apl": "aplikasi",
    "eror": "error",
    "good": "baik",
    "best": "bagus",
    "nice": "bagus",
    "mantul": "mantap",
    "mengecewakan": "kecewa",
    "ok": "bagus",
    "oke": "bagus",
    "baik": "bagus",
}
slang_dict.update(normalization_dict)


# Mengganti kata tidak baku dengan yang baku
def normalize_word(word, normalization_dict):
    return normalization_dict.get(word, word)


# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# Menangani Negasi
def handle_negation(tokens):
    negations = {
        "tidak",
        "tdk",
        "tidk",
        "tak",
        "tiada",
        "bukan",
        "gak",
        "ngga",
        "enggak",
        "nggak",
        "nggk",
        "ga",
        "gk",
        "ndak",
        "kurang",
    }
    new_tokens = []
    negate = False

    antonim_dict = {
        "bagus": "jelek",
        "keren": "jelek",
        "cakep": "jelek",
        "baik": "buruk",
        "mantap": "kacau",
        "kuat": "lemah",
        "bersih": "kotor",
        "terbaik": "terburuk",
        "cepat": "lambat",
        "kenceng": "lambat",
        "kencang": "lambat",
        "responsif": "lemot",
        "lancar": "macet",
        "instan": "lambat",
        "mudah": "sulit",
        "ringan": "berat",
        "enteng": "berat",
        "efisien": "boros",
        "stabil": "gagal",
        "inovatif": "usang",
        "modern": "kuno",
        "aman": "risiko",
        "praktis": "ribet",
        "membantu": "menghambat",
        "bantu": "hambat",
        "segera": "lambat",
        "sederhana": "rumit",
        "terhubung": "terputus",
        "optimal": "buruk",
        "jelek": "bagus",
        "buruk": "baik",
        "kacau": "mantap",
        "lemah": "kuat",
        "kotor": "bersih",
        "terburuk": "terbaik",
        "lambat": "cepat",
        "berat": "ringan",
        "boros": "efisien",
        "gagal": "stabil",
        "usang": "inovatif",
        "kuno": "modern",
        "risiko": "aman",
        "ribet": "praktis",
        "menghambat": "membantu",
        "hambat": "bantu",
        "rumit": "sederhana",
        "terputus": "terhubung",
        "lemot": "responsif",
        "macet": "lancar",
        "sulit": "mudah",
    }

    # for word in tokens:
    #    if word in negations:
    #        new_tokens.append("tidak_")
    #        negate = True  # Menambahkan kata negasi itu sendiri
    #    elif negate:
    #        # Jika sebelumnya adalah negasi, gabungkan dengan prefix 'tidak_'
    #        new_tokens[-1] = new_tokens[-1] + word  # Gabungkan dengan kata sebelumnya
    #        negate = False  # Reset flag negasi
    #    else:
    #        new_tokens.append(word)
    #
    # return new_tokens

    # for word in tokens:
    #    if word in negations:
    #        new_tokens.append(
    #            "tidak_"
    #        )  # Menambahkan prefix 'tidak_' untuk kata setelah negasi
    #        negate = True
    #    elif negate:
    #        antonim_word = antonim_dict.get(
    #            word, word
    #        )  # Mencari antonim dari kata setelah 'tidak_'
    #        new_tokens[-1] = (
    #            antonim_word  # Menghapus prefix 'tidak_' dan mengganti dengan antonim
    #        )
    #        negate = False  # Reset negasi setelah satu kata
    #    else:
    #        new_tokens.append(word)
    #
    # return new_tokens

    for word in tokens:
        if word in negations:
            new_tokens.append("tidak_")  # Menambahkan 'tidak_'
            negate = True
        elif negate:
            antonim_word = antonim_dict.get(word, None)
            if antonim_word:  # Mencari antonim dari kata setelah 'tidak_'
                new_tokens[-1] = antonim_word
            else:
                new_tokens[-1] = new_tokens[-1] + word
            negate = False
        else:
            new_tokens.append(word)

    return new_tokens


stop_words = set(stopwords.words("indonesian"))
custom_stopwords = {
    "terlalu",
    "banyak",
    "selama",
    "besar",
    "lama",
    "digunakan",
    "tidak",
    "cukup",
    "lumayan",
    "masalah",
    "bisa",
    "kendala",
    "livin",
    "bagus",
    "jelek",
    "mantap",
    "keren",
    "kurang",
    "baik",
    "lemot",
    "luar biasa",
    "sempurna",
    "top",
    "cepat",
    "responsif",
    "lancar",
    "ringan",
    "instan",
    "mudah",
    "simpel",
    "praktis",
    "nyaman",
    "lengkap",
    "inovatif",
    "canggih",
    "membantu",
    "berguna",
    "bermanfaat",
    "aman",
    "terpercaya",
    "stabil",
    "handal",
    "buruk",
    "parah",
    "tidak bagus",
    "tidak baik",
    "kecewa",
    "lambat",
    "loading",
    "macet",
    "responsif",
    "bug",
    "gagal",
    "crash",
    "keluar",
    "terbatas",
    "ribet",
    "susah",
    "aman",
    "rawan",
    "masalah",
}
stop_words = stop_words - custom_stopwords

keep_words = {
    "selama",
    "lama",
    "terasa",
    "livin",
    "info",
    "kendala",
    "masalah",
    "bisa",
    "bagus",
    "jelek",
    "mantap",
    "keren",
    "kurang",
    "baik",
    "lemot",
    "luar biasa",
    "sempurna",
    "top",
    "cepat",
    "responsif",
    "lancar",
    "ringan",
    "instan",
    "mudah",
    "simpel",
    "praktis",
    "nyaman",
    "lengkap",
    "inovatif",
    "canggih",
    "membantu",
    "berguna",
    "bermanfaat",
    "aman",
    "terpercaya",
    "stabil",
    "handal",
    "buruk",
    "parah",
    "kecewa",
    "lambat",
    "loading",
    "macet",
    "responsif",
    "bug",
    "gagal",
    "crash",
    "keluar",
    "terbatas",
    "ribet",
    "susah",
    "rawan",
    "rekomen",
    "tidak_bagus",
    "tidak_baik",
    "tidak_cepat",
    "tidak_responsif",
    "tidak_lancar",
    "tidak_membantu",
    "tidak_berguna",
    "tidak_nyaman",
    "tidak_mudah",
    "tidak_stabil",
    "tidak_handal",
    "tidak_aman",
    "tidak_terpercaya",
    "tidak_memuaskan",
    "tidak_mantap",
    "tidak_menarik",
    "tidak_keren",
    "tidak_canggih",
    "tidak_inovatif",
    "tidak_buruk",
    "tidak_lambat",
    "tidak_lemot",
    "tidak_ribet",
    "tidak_rumit",
    "tidak_gagal",
    "tidak_bug",
    "tidak_crash",
    "tidak_macet",
    "tidak_susah",
    "tidak_rawan",
    "tidak_kacau",
}


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    print(f"Original Text: {text}")

    # Cleansing (Menghapus karakter khusus dan angka)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    print(f"After Cleansing: {text}")

    # Case Folding (Menurunkan semua huruf menjadi huruf kecil)
    text = text.lower()
    print(f"After Case Folding: {text}")

    tokens = text.split()  # Tokenisasi dilakukan di sini
    print(f"Setelah Tokenisasi: {tokens}")

    # Word Normalization (Mengganti kata yang tidak baku dengan kata baku)
    tokens = [normalize_word(word, normalization_dict) for word in tokens]
    print(f"After Word Normalization: {tokens}")

    # Stemming (Mengubah kata menjadi bentuk dasar, kecuali kata yang ada dalam keep_words)
    tokens = [stemmer.stem(word) if word not in keep_words else word for word in tokens]
    print(f"After Stemming : {tokens}")

    # Handling Negation (Menangani negasi dengan menambahkan 'tidak_')
    tokens = handle_negation(tokens)
    print(f"After Handling Negation: {tokens}")

    # Stopword Removal (hapus kata tidak penting)
    tokens = [word for word in tokens if word not in stop_words]
    print(f"After Stopword Removal: {tokens}")

    return tokens


def predict_sentiment(input_text):
    word2vec_model = joblib.load("8Newword2vec_sg_model_vector300_window5_epochs5.pkl")
    rf = joblib.load("8Newrandom_forest_model_vector300_window5_epochs5.pkl")
    label_encoder = joblib.load("8Newlabel_encoder_vector300_window5_epochs5.pkl")

    processed_text = preprocess_text(input_text)

    input_vector = get_sentence_vector(processed_text, word2vec_model).reshape(1, -1)

    input_proba = rf.predict_proba(input_vector)[0]

    input_pred = np.argmax(input_proba)

    predicted_label = label_encoder.inverse_transform([input_pred])[0]

    class_labels = label_encoder.classes_

    # Mengonversi hasil prediksi kembali ke label asli (string)
    # hasil_prediksi = label_encoder.inverse_transform(input_pred)[0]
    # return hasil_prediksi

    print("\nProbabilitas prediksi:")
    for label, proba in zip(class_labels, input_proba):
        print(f"- {label.capitalize()}: {proba:.4f}")

    return predicted_label


if __name__ == "__main__":
    print("\n=== PREDIKSI TEKS BARU ===")
    while True:
        input_text = input("Teks (atau ketik 'exit' untuk keluar): ")
        if input_text.lower() == "exit":
            break
        result = predict_sentiment(input_text)
        print(f"Hasil prediksi untuk teks: '{input_text}' adalah : {result}\n")
