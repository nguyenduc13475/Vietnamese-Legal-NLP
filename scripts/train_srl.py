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
    AutoModel,
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
        # 11 classes in SRL_MAP
        loss_fct = LegalFocalLoss()
        loss = loss_fct(logits.view(-1, 11), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomSRLDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        ner_ids = [f.pop("ner_ids") for f in features]
        dep_ids = [f.pop("dep_ids") for f in features]
        p_ner_ids = [f.pop("p_ner_ids") for f in features]

        batch = super().__call__(features)
        max_len = batch["input_ids"].shape[1]

        batch["ner_ids"] = torch.tensor(
            [n + [0] * (max_len - len(n)) for n in ner_ids], dtype=torch.long
        )
        batch["dep_ids"] = torch.tensor(
            [d + [0] * (max_len - len(d)) for d in dep_ids], dtype=torch.long
        )
        batch["p_ner_ids"] = torch.tensor(
            [p + [0] * (max_len - len(p)) for p in p_ner_ids], dtype=torch.long
        )
        return batch


def train_srl(epochs, batch_size, lr):
    print("--- Training SRL using pre-annotated structural features ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    with open("data/annotated/srl_train.json", "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open("data/annotated/srl_test.json", "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    def prepare_data(examples):
        # We wrap input_ids with BOS/EOS to match auto_annotate.py logic
        # Input features are already aligned in JSON, just need to add BOS/EOS markers (ID 0)
        tokenized = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "ner_ids": [],
            "dep_ids": [],
            "p_ner_ids": [],
        }
        for i, ids in enumerate(examples["input_ids"]):
            input_ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            label_ids = (
                [-100] + [SRL2ID.get(t, 0) for t in examples["srl_tags"][i]] + [-100]
            )

            if len(input_ids) > 256:
                input_ids, label_ids = input_ids[:256], label_ids[:256]

            tokenized["input_ids"].append(input_ids)
            tokenized["attention_mask"].append([1] * len(input_ids))
            tokenized["labels"].append(label_ids)
            # Pad structural features with 0 for BOS/EOS
            tokenized["ner_ids"].append(([0] + examples["ner_ids"][i] + [0])[:256])
            tokenized["dep_ids"].append(([0] + examples["dep_ids"][i] + [0])[:256])
            tokenized["p_ner_ids"].append(([0] + examples["p_ner_ids"][i] + [0])[:256])
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

    sem_base = AutoModel.from_pretrained(BASE_MODEL_NAME)
    joint_model = JointSRLModel(sem_base)
    model = RobustSRLModel(joint_model)

    training_args = TrainingArguments(
        output_dir="./models/srl_checkpoints",
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    def compute_metrics(p):
        # Safely extract logits if Trainer wraps predictions in a tuple
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )
        preds = np.argmax(predictions, axis=2).flatten()
        labs = p.label_ids.flatten()

        mask = labs != -100
        return {
            "accuracy": float(accuracy_score(labs[mask], preds[mask])),
            "f1": float(
                f1_score(labs[mask], preds[mask], average="weighted", zero_division=0)
            ),
        }

    trainer = SRLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CustomSRLDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    out_dir = "models/srl"
    os.makedirs(out_dir, exist_ok=True)
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
