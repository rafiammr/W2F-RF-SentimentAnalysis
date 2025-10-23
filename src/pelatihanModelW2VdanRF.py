import time
import pandas as pd
import numpy as np
import ast
import multiprocessing
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
import joblib


def get_sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


if __name__ == "__main__":
    start_time = time.time()

    df = pd.read_csv("DatasetLivinPreproNewFinal.csv")
    df["text_cleaned"] = df["text_cleaned"].apply(ast.literal_eval)
    sentences_dataset = df["text_cleaned"].tolist()

    # Pelatihan Word2Vec
    print("Melatih Word2Vec...")

    vector_size = 200
    window = 20
    epochs = 5

    word2vec_start_time = time.time()
    word2vec_model = Word2Vec(
        sentences=sentences_dataset,
        vector_size=vector_size,
        window=window,
        epochs=epochs,
        min_count=2,
        sg=0,
        workers=multiprocessing.cpu_count() - 1,
    )

    print("\nParameter Model Word2Vec:")
    print(f"Vector Size: {vector_size}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")

    # Konversi kalimat ke vektor
    X = np.array(
        [get_sentence_vector(tokens, word2vec_model) for tokens in sentences_dataset]
    )
    y = df["sentiment"]

    # Label Encoding untuk y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pelatihan Random Forest
    print("Melatih Random Forest...")
    rf_start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        max_depth=10,
        min_samples_split=25,
    )
    rf.fit(X_train, y_train)

    y_pred_prob = rf.predict_proba(X_test)

    threshold = 0.4

    y_pred_adjusted = (y_pred_prob[:, 1] >= threshold).astype(int)

    # Evaluasi
    train_accuracy = accuracy_score(y_train, rf.predict(X_train))
    test_accuracy = accuracy_score(y_test, rf.predict(X_test))

    print(f"Akurasi model pada data latih: {train_accuracy * 100:.2f}%")
    print(f"Akurasi model pada data uji: {test_accuracy * 100:.2f}%")

    # Laporan Klasifikasi
    print("\nLaporan Klasifikasi:")
    # y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred_adjusted, digits=4))

    precision = precision_score(y_test, y_pred_adjusted, average=None, zero_division=1)
    recall = recall_score(y_test, y_pred_adjusted, average=None, zero_division=1)
    f1 = f1_score(y_test, y_pred_adjusted, average=None)

    for i, label in enumerate(np.unique(y_test)):
        print(f"Class {label}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-Score: {f1[i]:.4f}\n")

    print(
        f"F1-Score (Weighted): {f1_score(y_test, y_pred_adjusted, average='weighted'):.4f}"
    )

    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"Average CV score: {np.mean(cv_scores)}")

    word2vec_model_filename = (
        f"13Newword2vec_sg_model_vector{vector_size}_window{window}_epochs{epochs}.pkl"
    )
    rf_model_filename = f"13Newrandom_forest_model_vector{vector_size}_window{window}_epochs{epochs}.pkl"
    label_encoder_filename = (
        f"13Newlabel_encoder_vector{vector_size}_window{window}_epochs{epochs}.pkl"
    )
    joblib.dump(word2vec_model, word2vec_model_filename)
    joblib.dump(rf, rf_model_filename)
    joblib.dump(label_encoder, label_encoder_filename)

    print(f"Model Word2Vec disimpan dengan nama: {word2vec_model_filename}")
    print(f"Model Random Forest disimpan dengan nama: {rf_model_filename}")
    print(f"Model Label Encoder disimpan dengan nama:{label_encoder_filename}")

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\nTotal waktu eksekusi: {minutes} menit {seconds:.2f} detik")
