import argparse
import json
import os
import pickle

import numpy as np
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def train_tfidf():
    print("Training Intent Classifier (TF-IDF + Logistic Regression)...")
    train_path = "data/annotated/intent_train.json"
    test_path = "data/annotated/intent_test.json"

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    X_train = [item["text"] for item in train_data]
    y_train = [item["label"] for item in train_data]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_train_pred = model.predict(X_train_vec)
    print("\nClassification Report (Train Set):")
    print(classification_report(y_train, y_train_pred))

    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        X_test = [item["text"] for item in test_data]
        y_test = [item["label"] for item in test_data]

        X_test_vec = vectorizer.transform(X_test)
        y_test_pred = model.predict(X_test_vec)
        report = classification_report(y_test, y_test_pred, zero_division=0)
        print("\nClassification Report (Test Set):")
        print(report)

        os.makedirs("report", exist_ok=True)
        with open("report/intent_evaluation.txt", "w", encoding="utf-8") as f:
            f.write("=== Intent Classification Report (TF-IDF) ===\n")
            f.write(report)
        print("Report saved at: report/intent_evaluation.txt")

    os.makedirs("models/fine_tuned", exist_ok=True)
    with open("models/fine_tuned/intent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/fine_tuned/intent_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Saved TF-IDF model to models/fine_tuned/")


def train_transformer():
    print("Training Intent Classifier (PhoBERT)...")
    train_path = "data/annotated/intent_train.json"
    test_path = "data/annotated/intent_test.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    labels = sorted(list(set([item["label"] for item in train_data])))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    def format_data(data):
        return {
            "text": [item["text"] for item in data],
            "label": [label2id[item["label"]] for item in data],
        }

    train_dataset = Dataset.from_dict(format_data(train_data))
    test_dataset = Dataset.from_dict(format_data(test_data))

    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=256
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="models/fine_tuned_intent_transformer",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Detailed Evaluation for Classification Report
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Converting label IDs back to label names
    y_true = [id2label[label] for label in tokenized_test["label"]]
    y_pred = [id2label[p] for p in preds]

    report_str = classification_report(y_true, y_pred, zero_division=0)
    print("\nTransformer Classification Report:")
    print(report_str)

    os.makedirs("report", exist_ok=True)
    with open("report/intent_transformer_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("=== Intent Classification Report (Transformer - PhoBERT) ===\n")
        f.write(report_str)

    trainer.save_model("models/fine_tuned_intent_transformer")
    tokenizer.save_pretrained("models/fine_tuned_intent_transformer")
    print("Saved Transformer model to models/fine_tuned_intent_transformer/")
    print("Report saved at: report/intent_transformer_evaluation.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["tfidf", "transformer", "all"],
        help="Choose a model to train",
    )
    args = parser.parse_args()

    if args.model == "tfidf":
        train_tfidf()
    elif args.model == "transformer":
        train_transformer()
    elif args.model == "all":
        train_tfidf()
        train_transformer()
