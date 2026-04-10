import json
import os

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def evaluate():
    MODEL_PATH = "models/fine_tuned_intent_transformer"
    TEST_DATA = "data/annotated/intent_test.json"

    if not os.path.exists(MODEL_PATH):
        print("Model not trained!")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels_true = [item["label"] for item in data]

    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=256
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    labels_pred = [model.config.id2label[p] for p in preds]

    report = classification_report(labels_true, labels_pred, zero_division=0)
    print(report)

    os.makedirs("report", exist_ok=True)
    with open("report/intent_transformer_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("=== Intent Classification Report (Transformer) ===\n")
        f.write(report)


if __name__ == "__main__":
    evaluate()
