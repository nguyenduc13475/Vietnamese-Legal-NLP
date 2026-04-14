import json
import os
import sys

import numpy as np
import torch
from seqeval.metrics import classification_report
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_ner import RobustNERModel


def evaluate_ner():
    MODEL_PATH = "./models/ultra_ner"
    BASE_MODEL_NAME = "vinai/phobert-base"
    TEST_DATA_PATH = "data/annotated/ner_test.json"

    if not os.path.exists(MODEL_PATH):
        print(
            f"Error: Model not found at {MODEL_PATH}. Please run 'make train-ner' first."
        )
        return

    print("--- Initializing Robust Architecture ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    id2label = {
        0: "O",
        1: "B-PARTY",
        2: "I-PARTY",
        3: "B-MONEY",
        4: "I-MONEY",
        5: "B-DATE",
        6: "I-DATE",
        7: "B-RATE",
        8: "I-RATE",
        9: "B-PENALTY",
        10: "I-PENALTY",
        11: "B-LAW",
        12: "I-LAW",
        13: "B-OBJECT",
        14: "I-OBJECT",
        15: "B-PREDICATE",
        16: "I-PREDICATE",
    }
    label2id = {v: k for k, v in id2label.items()}

    config = AutoConfig.from_pretrained(
        BASE_MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    # 1. Initialize base model with fine-tuned config
    raw_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )

    # 2. Wrap into Robust architecture
    model = RobustNERModel(raw_model)

    # 3. Load weights with prefix handling (RobustNERModel wraps under 'base_model')
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")

    # Compatibility fix: if keys don't start with base_model, they might be from a raw trainer save
    first_key = list(state_dict.keys())[0]
    if not first_key.startswith("base_model."):
        print("Mapping state_dict keys to Robust wrapper...")
        state_dict = {f"base_model.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    id_to_label = config.id2label
    y_true = []
    y_pred = []

    print(f"Evaluating {len(test_data)} samples on {device}...")

    for item in tqdm(test_data, desc="Inference"):
        # Add BOS/EOS to match training preprocessing
        original_ids = item["input_ids"]
        input_ids_batched = (
            [tokenizer.bos_token_id] + original_ids + [tokenizer.eos_token_id]
        )

        # Truncate if necessary (match training max_length)
        if len(input_ids_batched) > 256:
            input_ids_batched = input_ids_batched[:256]

        input_tensor = torch.tensor([input_ids_batched]).to(device)
        attn_mask = torch.tensor([[1] * len(input_ids_batched)]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask)
            logits = outputs["logits"]

        # Get predictions and strip BOS/EOS positions
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        # Predictions are [BOS, tok1, tok2, ..., tokN, EOS]
        # Labels in ner_test.json are only for [tok1, ..., tokN]
        pred_labels = [
            id_to_label[p] if isinstance(p, (int, np.integer)) else str(p)
            for p in predictions
        ]

        # Strip BOS (index 0) and EOS (last index)
        meaningful_preds = pred_labels[1:-1]

        # Sync lengths
        true_tags = item["ner_tags"]
        min_len = min(len(true_tags), len(meaningful_preds))

        y_true.append(true_tags[:min_len])
        y_pred.append(meaningful_preds[:min_len])

    report = classification_report(y_true, y_pred, digits=4)
    print("\n" + "=" * 50 + "\nNER EVALUATION REPORT\n" + "=" * 50)
    print(report)

    os.makedirs("report", exist_ok=True)
    with open("report/ner_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Full report saved to report/ner_evaluation.txt")


if __name__ == "__main__":
    evaluate_ner()
