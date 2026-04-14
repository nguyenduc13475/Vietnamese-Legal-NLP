import json
import os

import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


def evaluate_segmenter():
    MODEL_PATH = "models/fine_tuned_segmenter"
    TEST_DATA_PATH = "data/annotated/segment_test.json"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        return

    print("--- Evaluating Segmenter Model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    y_true = []
    y_pred = []

    for item in tqdm(test_data, desc="Inference"):
        input_ids = (
            [tokenizer.bos_token_id] + item["input_ids"] + [tokenizer.eos_token_id]
        )
        input_tensor = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits

        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        # Cắt bỏ BOS và EOS để khớp với label gốc
        meaningful_preds = predictions[1:-1]
        true_tags = item["segment_tags"]

        y_true.extend(true_tags)
        y_pred.extend(meaningful_preds[: len(true_tags)])

    # Label 0-5 tương ứng với các thành phần mệnh đề
    target_names = [f"Component_{i}" for i in range(6)]
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    print("\n" + report)

    os.makedirs("report", exist_ok=True)
    with open("report/segmenter_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("=== Segmenter Evaluation Report ===\n")
        f.write(report)
    print("Report saved to report/segmenter_evaluation.txt")


if __name__ == "__main__":
    evaluate_segmenter()
