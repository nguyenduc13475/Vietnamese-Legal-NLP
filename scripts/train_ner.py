import argparse
import json
import os

import numpy as np
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def load_custom_data(file_path):
    """
    Loading NER training data from JSON.
    NER tag format: 0: O, 1: B-PARTY, 2: I-PARTY, 3: B-MONEY, 4: I-MONEY...
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data not found at {file_path}. Please generate data before training."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def train(model_name: str, epochs: int, batch_size: int, learning_rate: float):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the label map to be saved directly into the model config
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
    }
    label2id = {v: k for k, v in id2label.items()}

    num_labels = 13
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    train_dataset = load_custom_data("data/annotated/ner_train.json")
    eval_dataset = load_custom_data("data/annotated/ner_test.json")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for i, words in enumerate(examples["tokens"]):
            ner_tags = examples["ner_tags"][i]

            input_ids = [tokenizer.cls_token_id]
            label_ids = [-100]

            for word, label in zip(words, ner_tags):
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                if not word_tokens:
                    continue

                input_ids.extend(word_tokens)

                for j in range(len(word_tokens)):
                    if j == 0:
                        label_ids.append(label)
                    else:
                        if label == 0:
                            label_ids.append(0)
                        elif label % 2 != 0:
                            label_ids.append(label + 1)
                        else:
                            label_ids.append(label)

            input_ids.append(tokenizer.sep_token_id)
            label_ids.append(-100)

            if len(input_ids) > 256:
                input_ids = input_ids[:255] + [tokenizer.sep_token_id]
                label_ids = label_ids[:255] + [-100]

            tokenized_inputs["input_ids"].append(input_ids)
            tokenized_inputs["attention_mask"].append([1] * len(input_ids))
            tokenized_inputs["labels"].append(label_ids)

        return tokenized_inputs

    # Remove the original string columns to prevent PyTorch tensor collation crashes
    tokenized_datasets = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_eval_datasets = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir="./models/fine_tuned_ner",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, la) in zip(prediction, label) if la != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[la] for (_, la) in zip(prediction, label) if la != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_eval_datasets,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print(f"Starting NER model training using {model_name}...")
    trainer.train()

    output_dir = "./models/fine_tuned_ner"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done! Model saved at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER Model for Legal Contracts")
    parser.add_argument(
        "--model",
        type=str,
        default="vinai/phobert-base",
        help="Pretrained model (e.g., vinai/phobert-large)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
