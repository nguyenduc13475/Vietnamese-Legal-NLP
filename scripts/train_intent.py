import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from underthesea import word_tokenize

# --- CONSISTENCY LAYER: Focal Loss & Robust Architecture ---


class LegalFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        ce_loss = -(true_dist * log_probs).sum(dim=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class RobustIntentModel(nn.Module):
    """Wrapper to add Multi-Sample Dropout for stable intent classification."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model.roberta(
            input_ids, attention_mask=attention_mask, return_dict=True
        )
        # Use the first token [CLS] as the pooled representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(pooled_output))
        logits /= len(self.dropouts)

        return {"logits": logits}


class StableTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = LegalFocalLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_tfidf():
    print("Training Intent Classifier (TF-IDF + Logistic Regression)...")
    train_path = "data/annotated/intent_train.json"
    test_path = "data/annotated/intent_test.json"

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Use word_tokenize to make TF-IDF features match PhoBERT style
    X_train = [word_tokenize(item["text"], format="text") for item in train_data]
    y_train = [item["label"] for item in train_data]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_train_pred = model.predict(X_train_vec)
    print("\nClassification Report (Train Set):")
    print(classification_report(y_train, y_train_pred))

    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        X_test = [word_tokenize(item["text"], format="text") for item in test_data]
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
    print(f"Starting Robust Training for Intent ({model_name})...")
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
            # Standardize with underthesea before tokenizing
            "text": [word_tokenize(item["text"], format="text") for item in data],
            "labels": [label2id[item["label"]] for item in data],
        }

    train_dataset = Dataset.from_dict(format_data(train_data))
    test_dataset = Dataset.from_dict(format_data(test_data))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_test = test_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    raw_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    # Apply Robust Wrapper
    model = RobustIntentModel(raw_model)

    training_args = TrainingArguments(
        output_dir="models/intent_checkpoints",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        }

    trainer = StableTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    out_dir = "models/fine_tuned_intent_transformer"
    os.makedirs(out_dir, exist_ok=True)

    # Save the base model's state for easier loading in inference
    torch.save(
        model.base_model.state_dict(), os.path.join(out_dir, "pytorch_model.bin")
    )
    tokenizer.save_pretrained(out_dir)
    raw_model.config.save_pretrained(out_dir)

    print(f"Success! Robust Intent model saved to {out_dir}")


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
