import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Ensure root is in path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extraction.srl_engine import (
    BASE_MODEL_NAME,
    SRL2ID,
    JointSRLModel,
    RobustSRLModel,
)


class LegalFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = torch.log_softmax(inputs, dim=-1)
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


class SRLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = LegalFocalLoss()
        loss = loss_fct(logits.view(-1, 11), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_srl(epochs, batch_size, lr):
    print("--- Starting SRL Training (Robust Architecture) ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    train_path = "data/annotated/srl_train.json"
    test_path = "data/annotated/srl_test.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    def prepare_data(examples):
        tokenized = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "ner_ids": [],
            "dep_ids": [],
            "p_ner_ids": [],
        }
        for i, ids in enumerate(examples["input_ids"]):
            tags = [SRL2ID.get(t, 0) for t in examples["srl_tags"][i]]
            # Add BOS/EOS to match auto_annotate logic
            input_ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            label_ids = [-100] + tags + [-100]

            if len(input_ids) > 256:
                input_ids, label_ids = input_ids[:256], label_ids[:256]

            seq_len = len(input_ids)
            tokenized["input_ids"].append(input_ids)
            tokenized["attention_mask"].append([1] * seq_len)
            tokenized["labels"].append(label_ids)
            # Placeholder for O2 features during training as they are not in JSON
            tokenized["ner_ids"].append([0] * seq_len)
            tokenized["dep_ids"].append([0] * seq_len)
            tokenized["p_ner_ids"].append([0] * seq_len)
        return tokenized

    train_ds = Dataset.from_list(train_raw).map(
        prepare_data,
        batched=True,
        remove_columns=Dataset.from_list(train_raw).column_names,
    )
    eval_ds = Dataset.from_list(test_raw).map(
        prepare_data,
        batched=True,
        remove_columns=Dataset.from_list(test_raw).column_names,
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        y_true = labels.flatten()
        y_pred = predictions.flatten()
        mask = y_true != -100
        return {
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "f1": f1_score(y_true[mask], y_pred[mask], average="weighted"),
        }

    joint_model = JointSRLModel()
    model = RobustSRLModel(joint_model)

    training_args = TrainingArguments(
        output_dir="./models/srl_checkpoints",
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
    )

    trainer = SRLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    out_dir = "models/ultra_srl"
    os.makedirs(out_dir, exist_ok=True)
    # Save the base model's state dict for engine compatibility
    torch.save(
        model.base_model.state_dict(), os.path.join(out_dir, "pytorch_model.bin")
    )
    tokenizer.save_pretrained(out_dir)
    print(f"Success! SRL Model saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    train_srl(args.epochs, args.batch_size, args.lr)
