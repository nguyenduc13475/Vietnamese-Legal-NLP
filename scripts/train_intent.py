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
    DataCollatorWithPadding,
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


def train_transformer(
    model_name: str, epochs: int, batch_size: int, learning_rate: float
):
    print(f"Training Intent Classifier ({model_name})...")
    train_path = "data/annotated/intent_train.json"
    test_path = "data/annotated/intent_test.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    labels = sorted(list(set([item["label"] for item in train_data])))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # Renamed "label" to "labels" to match PyTorch/HuggingFace expected input
    def format_data(data):
        return {
            "text": [item["text"] for item in data],
            "labels": [label2id[item["label"]] for item in data],
        }

    train_dataset = Dataset.from_dict(format_data(train_data))
    test_dataset = Dataset.from_dict(format_data(test_data))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use dynamic padding instead of fixed max_length to save GPU memory
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    # Remove the string "text" column to prevent tensor collation errors
    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_test = test_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="models/fine_tuned_intent_transformer",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Explicitly look for the best F1 score
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
        data_collator=data_collator,
    )

    trainer.train()

    # Detailed Evaluation for Classification Report
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Converting label IDs back to label names
    y_true = [id2label[label] for label in tokenized_test["labels"]]
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
        "--model", type=str, default="all", choices=["tfidf", "transformer", "all"]
    )
    parser.add_argument(
        "--transformer_model",
        type=str,
        default="vinai/phobert-base",
        help="Pretrained model base",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for Transformer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for Transformer"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate for Transformer"
    )
    args = parser.parse_args()

    if args.model in ["tfidf", "all"]:
        train_tfidf()
    if args.model in ["transformer", "all"]:
        train_transformer(
            model_name=args.transformer_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
