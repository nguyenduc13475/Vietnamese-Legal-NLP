import json
import os
import sys

import numpy as np
import torch
from seqeval.metrics import classification_report
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

# Fix path to allow importing from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_ner import RobustNERModel


def evaluate_ner():
    MODEL_PATH = "./models/ultra_ner"
    # We need the original base model name to initialize the architecture
    BASE_MODEL_NAME = "vinai/phobert-base"
    TEST_DATA_PATH = "data/annotated/ner_test.json"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Loading Robust Model architecture and weights...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 1. Load the Config from the saved folder
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # 2. Initialize the standard model with this config
    # We use BASE_MODEL_NAME to ensure the 'model_type' is recognized,
    # but the config will have your custom label mappings.
    raw_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )

    # 3. Wrap it
    model = RobustNERModel(raw_model)

    # 4. Load the saved weights (state_dict) manually
    # Trainer.save_model saves the state_dict in pytorch_model.bin
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        # If the state_dict was saved via the wrapper, keys start with 'base_model.'
        # If saved via Trainer on the wrapped model, we load it into the wrapper
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: pytorch_model.bin not found in {MODEL_PATH}. Using raw weights."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    id_to_label = model.config.id2label
    y_true = []
    y_pred = []

    print(f"Evaluating {len(test_data)} samples...")

    for item in test_data:
        input_ids = torch.tensor([item["input_ids"]]).to(device)
        attention_mask = torch.tensor([[1] * len(item["input_ids"])]).to(device)
        true_tags = item["ner_tags"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        current_pred = [
            id_to_label[p] if isinstance(p, (int, np.integer)) else p
            for p in predictions
        ]

        min_len = min(len(true_tags), len(current_pred))
        y_true.append(true_tags[:min_len])
        y_pred.append(current_pred[:min_len])

    report = classification_report(y_true, y_pred, digits=4)
    print("\n=== Final Robust NER Evaluation Report ===")
    print(report)

    os.makedirs("report", exist_ok=True)
    with open("report/ner_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    evaluate_ner()
