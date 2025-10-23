import joblib
import numpy as np
import pandas as pd
import nltk
import re
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize the Flask app
app = Flask(__name__)

nltk.download("punkt")
nltk.download("stopwords")

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

word2vec_model = joblib.load("13Newword2vec_sg_model_vector200_window15_epochs5.pkl")
rf = joblib.load("13Newrandom_forest_model_vector200_window15_epochs5.pkl")
label_encoder = joblib.load("13Newlabel_encoder_vector200_window15_epochs5.pkl")


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
        return ""

    preprocessing_steps = {}

    print(f"Teks Asli: {text}")

    cleansing = re.sub(r"[^a-zA-Z\s]", " ", text)
    preprocessing_steps["cleansing"] = cleansing
    print(f"Setelah Cleansing: {cleansing}")

    case_folding = cleansing.lower()
    preprocessing_steps["case folding"] = case_folding
    print(f"Setelah Case Folding: {case_folding}")

    tokens = case_folding.split()
    preprocessing_steps["tokenization"] = tokens
    print(f"Setelah Tokenisasi: {tokens}")

    normalized = [normalize_word(word) for word in tokens]
    preprocessing_steps["word normalization"] = normalized
    print(f"Setelah Word Normalization: {normalized}")

    stemming = [
        stemmer.stem(word) if word not in keep_words else word for word in normalized
    ]
    preprocessing_steps["stemming"] = stemming
    print(f"Setelah Stemming : {stemming}")

    negation = handle_negation(stemming)
    preprocessing_steps["negation handling"] = negation
    print(f"Setelah Penanganan Negasi: {negation}")

    stopword = [word for word in negation if word not in stop_words]
    preprocessing_steps["stopword removal"] = stopword
    print(f"Setelah Penghapusan Stopword: {stopword}")

    return stopword, preprocessing_steps


def get_sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def predict_sentiment(input_text):
    processed_text, preprocessing_steps = preprocess_text(input_text)
    input_vector = get_sentence_vector(processed_text, word2vec_model).reshape(1, -1)
    input_proba = rf.predict_proba(input_vector)[0]
    input_pred = np.argmax(input_proba)
    predicted_label = label_encoder.inverse_transform([input_pred])[0]

    return predicted_label, input_proba, preprocessing_steps


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form["input_text"]
        sentiment, probabilities, preprocessing_steps = predict_sentiment(input_text)

        print(f"Sentiment: {sentiment}")
        print(f"Probabilities: {probabilities}")
        print(f"Preprocessing Steps: {preprocessing_steps}")

        class_labels = label_encoder.classes_
        prob_result = {
            label.capitalize(): prob for label, prob in zip(class_labels, probabilities)
        }

        return render_template(
            "index.html",
            input_text=input_text,
            sentiment=sentiment,
            prob_result=prob_result,
            preprocessing_steps=preprocessing_steps,
        )

    return render_template("index.html", sentiment=None)


if __name__ == "__main__":
    app.run(debug=True)
