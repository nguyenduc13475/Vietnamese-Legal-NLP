import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize


# Re-define wrapper for loading
class RobustIntentModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.base_model.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(pooled_output))
        logits /= len(self.dropouts)
        return {"logits": logits}


def evaluate():
    MODEL_PATH = "models/fine_tuned_intent_transformer"
    BASE_MODEL = "vinai/phobert-base"
    TEST_DATA = "data/annotated/intent_test.json"

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}!")
        return

    print("--- Evaluating Robust Intent Transformer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # 1. Load weights manually because of the wrapper
    raw_model = AutoModelForSequenceClassification.from_config(config)
    model = RobustIntentModel(raw_model)

    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")
    # If the saved state_dict is from the base_model (as done in updated train_intent.py)
    model.base_model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Segment text to match training
    texts = [word_tokenize(item["text"], format="text") for item in data]
    labels_true = [item["label"] for item in data]

    y_pred = []

    # Process in small batches to avoid OOM
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
            y_pred.extend([config.id2label[p] for p in preds])

    report = classification_report(labels_true, y_pred, zero_division=0, digits=4)
    print(report)

    os.makedirs("report", exist_ok=True)
    with open("report/intent_transformer_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("=== Intent Classification Report (Robust Transformer) ===\n")
        f.write(report)
    print("Full report saved to report/intent_transformer_evaluation.txt")


if __name__ == "__main__":
    evaluate()
