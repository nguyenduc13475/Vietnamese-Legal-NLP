import json
import os
import sys

import torch
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Fix path to allow importing from scripts even when running inside the folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_ner import RobustNERModel


def evaluate_ner():
    MODEL_PATH = "./models/ultra_ner"
    TEST_DATA_PATH = "data/annotated/ner_test.json"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Loading robust model for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 1. Load the raw transformer first
    raw_model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    # 2. Wrap it in the Robust architecture used during training
    model = RobustNERModel(raw_model)

    # Load state dict (usually saved by trainer.save_model)
    # If using Trainer.save_model, weights are already inside raw_model.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}")
        return

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    id_to_label = model.config.id2label
    y_true = []
    y_pred = []

    print(f"Evaluating {len(test_data)} samples...")

    for item in test_data:
        # Use the input_ids directly from the JSON to ensure no re-tokenization noise
        input_ids = torch.tensor([item["input_ids"]]).to(device)
        attention_mask = torch.tensor([[1] * len(item["input_ids"])]).to(device)

        # Gold labels (ignore -100 if present, though usually not in raw test json)
        # Convert string labels to their actual tags if stored as strings
        true_tags = item["ner_tags"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        # Align predictions with gold tags
        # Logic: input_ids in JSON don't include BOS/EOS, but your training script added them.
        # However, if your JSON 'ner_tags' matches 'input_ids' length, we map 1:1.

        current_pred = [id_to_label[p] for p in predictions]

        # Ensure lengths match (trimming if there was any truncation during inference)
        min_len = min(len(true_tags), len(current_pred))

        y_true.append(true_tags[:min_len])
        y_pred.append(current_pred[:min_len])

    report = classification_report(y_true, y_pred, digits=4)
    print("\n=== Final Robust NER Evaluation Report ===")
    print(report)

    os.makedirs("report", exist_ok=True)
    with open("report/ner_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("=== Final Robust NER Evaluation Report (Seqeval) ===\n")
        f.write(report)
    print("Report saved to report/ner_evaluation.txt")


if __name__ == "__main__":
    evaluate_ner()
