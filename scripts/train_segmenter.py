import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


# Focal Loss: Penalizes missing hard-to-find segment boundaries
class LegalFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            ignore_mask = targets == -100
            targets_safe = targets.clone()
            targets_safe[ignore_mask] = 0
            true_dist.scatter_(1, targets_safe.unsqueeze(1), 1.0 - self.smoothing)

        ce_loss = -(true_dist * log_probs).sum(dim=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss[~ignore_mask].mean()


class RobustSegmenterModel(nn.Module):
    """Wrapper to add Multi-Sample Dropout for stable clause boundary detection."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        # Standard PhoBERT hidden size is 768
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Average the logits from 5 different dropout masks
        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(sequence_output))
        logits /= len(self.dropouts)

        return {"logits": logits}


class StableTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = LegalFocalLoss(smoothing=0.1)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_segmenter(model_name, epochs, batch_size, learning_rate):
    print(f"Starting Optimized Segmenter Training for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load data pre-processed by auto_annotate.py (Keys: input_ids, segment_tags)
    train_path = "data/annotated/segment_train.json"
    test_path = "data/annotated/segment_test.json"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Missing annotated data at {train_path}. Run auto_annotate first."
        )

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    def prepare_datasets(examples):
        tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
        for i, ids in enumerate(examples["input_ids"]):
            tag_list = examples["segment_tags"][i]

            # PhoBERT specific: Add BOS/EOS tokens
            input_ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            label_ids = [-100] + tag_list + [-100]

            if len(input_ids) > 256:
                input_ids, label_ids = input_ids[:256], label_ids[:256]

            tokenized["input_ids"].append(input_ids)
            tokenized["attention_mask"].append([1] * len(input_ids))
            tokenized["labels"].append(label_ids)
        return tokenized

    train_ds = Dataset.from_list(train_raw).map(
        prepare_datasets, batched=True, remove_columns=["input_ids", "segment_tags"]
    )
    eval_ds = Dataset.from_list(test_raw).map(
        prepare_datasets, batched=True, remove_columns=["input_ids", "segment_tags"]
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Flatten and filter out ignore index (-100)
        y_true = labels.flatten()
        y_pred = predictions.flatten()
        mask = y_true != -100

        return {
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "f1": f1_score(y_true[mask], y_pred[mask], average="weighted"),
        }

    raw_model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=6
    )
    # Synchronize config for pipeline compatibility
    raw_model.config.id2label = {i: str(i) for i in range(6)}
    raw_model.config.label2id = {str(i): i for i in range(6)}

    # Apply Robust Wrapper
    model = RobustSegmenterModel(raw_model)

    training_args = TrainingArguments(
        output_dir="./models/segmenter_checkpoints",
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

    trainer = StableTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    out_dir = "models/segmenter"
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Success! Optimized segmenter saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    train_segmenter("vinai/phobert-base", args.epochs, args.batch_size, args.lr)
