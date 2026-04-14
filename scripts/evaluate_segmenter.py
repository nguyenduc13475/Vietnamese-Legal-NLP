import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


# Import Robust architecture (must match train_segmenter.py)
class RobustSegmenterModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(sequence_output))
        logits /= len(self.dropouts)
        return {"logits": logits}


def evaluate_segmenter():
    MODEL_PATH = "models/fine_tuned_segmenter"
    BASE_MODEL_NAME = "vinai/phobert-base"
    TEST_DATA_PATH = "data/annotated/segment_test.json"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        return

    print("--- Initializing Robust Segmenter for Evaluation ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Define labels (0-5 for components)
    id2label = {i: f"Component_{i}" for i in range(6)}
    label2id = {v: k for k, v in id2label.items()}

    config = AutoConfig.from_pretrained(
        BASE_MODEL_NAME, num_labels=6, id2label=id2label, label2id=label2id
    )

    # 1. Initialize raw base model
    raw_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME, config=config, ignore_mismatched_sizes=True
    )

    # 2. Wrap into Robust architecture
    model = RobustSegmenterModel(raw_model)

    # 3. Load fine-tuned weights
    import safetensors.torch

    weights_safe = os.path.join(MODEL_PATH, "model.safetensors")
    weights_bin = os.path.join(MODEL_PATH, "pytorch_model.bin")

    if os.path.exists(weights_safe):
        state_dict = safetensors.torch.load_file(weights_safe)
    elif os.path.exists(weights_bin):
        state_dict = torch.load(weights_bin, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights found in {MODEL_PATH}")

    # Map keys if they were saved from raw trainer without the 'base_model.' prefix
    first_key = list(state_dict.keys())[0]
    if not first_key.startswith("base_model."):
        state_dict = {f"base_model.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    y_true = []
    y_pred = []

    print(f"Evaluating {len(test_data)} samples on {device}...")
    for item in tqdm(test_data, desc="Inference"):
        # Match training preprocessing: [BOS] + ids + [EOS]
        original_ids = item["input_ids"]
        input_ids_batched = (
            [tokenizer.bos_token_id] + original_ids + [tokenizer.eos_token_id]
        )

        if len(input_ids_batched) > 256:
            input_ids_batched = input_ids_batched[:256]

        input_tensor = torch.tensor([input_ids_batched]).to(device)
        attn_mask = torch.tensor([[1] * len(input_ids_batched)]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attn_mask)
            logits = outputs["logits"]

        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        # Strip BOS/EOS positions to align with 'segment_tags'
        meaningful_preds = predictions[1:-1]
        true_tags = item["segment_tags"]

        # Sync lengths in case of truncation
        min_len = min(len(true_tags), len(meaningful_preds))
        y_true.extend(true_tags[:min_len])
        y_pred.extend(meaningful_preds[:min_len])

    report = classification_report(
        y_true, y_pred, target_names=list(id2label.values()), zero_division=0, digits=4
    )
    print("\n" + "=" * 50 + "\nSEGMENTER EVALUATION REPORT\n" + "=" * 50)
    print(report)

    os.makedirs("report", exist_ok=True)
    with open("report/segmenter_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Full report saved to report/segmenter_evaluation.txt")


if __name__ == "__main__":
    evaluate_segmenter()
