import json
import os

import torch
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer


def evaluate_ner():
    MODEL_PATH = "./models/fine_tuned_ner"
    if not os.path.exists(MODEL_PATH):
        print("Mô hình NER chưa được train. Hãy chạy 'make train-ner' trước.")
        return

    print("Loading model and tokenizer for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

    test_file_path = "data/annotated/ner_test.json"
    if not os.path.exists(test_file_path):
        print(f"File {test_file_path} not found.")
        return

    with open(test_file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Convert keys from string (JSON default) to int to match the integer tags
    id_to_label = {int(k): v for k, v in model.config.id2label.items()}

    y_true = []
    y_pred = []

    print("Evaluating directly on the Tokenized Dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for item in test_data:
        words = item["tokens"]
        true_tags = [id_to_label[tag] for tag in item["ner_tags"]]
        y_true.append(true_tags)

        # Extracting predictions by aligning word_ids
        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        word_ids = inputs.word_ids(batch_index=0)

        # Moving tensors to the correct device (CPU/GPU)
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs_on_device)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        pred_tags = []
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                pred_tags.append(id_to_label[predictions[i]])
            previous_word_idx = word_idx

        y_pred.append(pred_tags)

        # Trimming y_true to ensure the length always matches y_pred
        # (handling cases where the Tokenizer truncates sequences when > max_length).
        y_true[-1] = y_true[-1][: len(pred_tags)]

    report = classification_report(y_true, y_pred)
    print("\n=== NER Evaluation Report (Seqeval) ===")
    print(report)

    os.makedirs("report", exist_ok=True)
    with open("report/ner_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("=== NER Evaluation Report (Seqeval) ===\n")
        f.write(report)
    print("Evaluation report saved at: report/ner_evaluation.txt")


if __name__ == "__main__":
    evaluate_ner()
