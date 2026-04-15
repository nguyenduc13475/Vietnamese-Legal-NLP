import json
import os
import sys

import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extraction.srl_engine import (
    BASE_MODEL_NAME,
    ID2SRL,
    MODEL_PATH,
    SRL2ID,
)
from src.models.robust_base import JointSRLModel, RobustSRLModel


def evaluate_srl():
    TEST_DATA_PATH = "data/annotated/srl_test.json"
    REPORT_PATH = "report/srl_evaluation.txt"

    if not os.path.exists(MODEL_PATH):
        print(
            f"Error: SRL Model not found at {MODEL_PATH}. Please run 'make train-srl' first."
        )
        return

    print("--- Initializing SRL Model for Evaluation ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Reconstruct the model architecture with the semantic base
    config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
    sem_base = AutoModel.from_config(config)
    base_joint = JointSRLModel(sem_base)
    model = RobustSRLModel(base_joint)

    # Load weights
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if os.path.exists(weights_path):
        # The engine saves the base_model's state_dict, but since we wrap it in RobustSRLModel
        # during inference/eval, we load it into model.base_model
        model.base_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(f"Error: Weights file not found at {weights_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loading test data from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    y_true = []
    y_pred = []

    print(f"Evaluating {len(test_data)} samples on {device}...")

    for item in tqdm(test_data, desc="SRL Inference"):
        original_ids = item["input_ids"]

        # Match training preprocessing: [BOS] + ids + [EOS]
        input_ids_batched = (
            [tokenizer.bos_token_id] + original_ids + [tokenizer.eos_token_id]
        )

        # We must load the real structural features from the test data and pad with 0 for BOS/EOS
        feat_ner = [0] + item["ner_ids"] + [0]
        feat_dep = [0] + item["dep_ids"] + [0]
        feat_p_ner = [0] + item["p_ner_ids"] + [0]

        if len(input_ids_batched) > 256:
            input_ids_batched = input_ids_batched[:256]
            feat_ner = feat_ner[:256]
            feat_dep = feat_dep[:256]
            feat_p_ner = feat_p_ner[:256]

        input_tensor = torch.tensor([input_ids_batched]).to(device)
        attn_mask = torch.tensor([[1] * len(input_ids_batched)]).to(device)
        ner_tensor = torch.tensor([feat_ner], dtype=torch.long).to(device)
        dep_tensor = torch.tensor([feat_dep], dtype=torch.long).to(device)
        p_ner_tensor = torch.tensor([feat_p_ner], dtype=torch.long).to(device)

        with torch.no_grad():
            # Feed the real structural features instead of zeros
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                ner_ids=ner_tensor,
                dep_ids=dep_tensor,
                p_ner_ids=p_ner_tensor,
            )
            logits = outputs["logits"]

        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        # Strip BOS/EOS positions to align with 'srl_tags'
        meaningful_preds = predictions[1:-1]
        true_tags_str = item["srl_tags"]

        # Convert true string tags to IDs for comparison
        true_tags = [SRL2ID.get(t, 0) for t in true_tags_str]

        # Sync lengths
        min_len = min(len(true_tags), len(meaningful_preds))
        y_true.extend(true_tags[:min_len])
        y_pred.extend(meaningful_preds[:min_len])

    # Generate Report
    target_names = [ID2SRL[i] for i in range(len(ID2SRL))]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(ID2SRL))),
        target_names=target_names,
        zero_division=0,
        digits=4,
    )

    print("\n" + "=" * 50 + "\nSRL EVALUATION REPORT\n" + "=" * 50)
    print(report)

    os.makedirs("report", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== Semantic Role Labeling (SRL) Evaluation Report ===\n")
        f.write("Model: Robust PhoBERT + Structural Embeddings\n")
        f.write(report)
    print(f"Full report saved to {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_srl()
