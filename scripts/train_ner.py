import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# Focal Loss: Penalizes missing hard-to-find legal entities like PENALTY or RATE
class LegalFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", weight=self.weight, ignore_index=-100
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class StableTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Use Focal Loss instead of standard CrossEntropy for better F1
        loss_fct = LegalFocalLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_custom_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def train(model_name, epochs, batch_size, learning_rate):
    print(f"Starting Optimized Training for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    train_ds = load_custom_data("data/annotated/ner_train.json")
    test_ds = load_custom_data("data/annotated/ner_test.json")

    def align_labels(examples):
        tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
        for i, ids in enumerate(examples["input_ids"]):
            tag_list = [
                label2id[t] if isinstance(t, str) else t
                for t in examples["ner_tags"][i]
            ]
            input_ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            label_ids = [-100] + tag_list + [-100]

            if len(input_ids) > 256:
                input_ids, label_ids = input_ids[:256], label_ids[:256]

            tokenized["input_ids"].append(input_ids)
            tokenized["attention_mask"].append([1] * len(input_ids))
            tokenized["labels"].append(label_ids)
        return tokenized

    train_dataset = train_ds.map(
        align_labels, batched=True, remove_columns=train_ds.column_names
    )
    eval_dataset = test_ds.map(
        align_labels, batched=True, remove_columns=test_ds.column_names
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./models/ultra_ner_checkpoints",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2label[p] for (p, la) in zip(pr, l) if la != -100]
            for pr, l in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[la] for (p, la) in zip(pr, l) if la != -100]
            for pr, l in zip(predictions, labels)
        ]
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    trainer = StableTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    out_dir = "./models/ultra_ner"
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Success! Optimized model saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train("vinai/phobert-base", args.epochs, 16, 3e-5)
