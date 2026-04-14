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
    Trainer,
    TrainingArguments,
)


# Focal Loss: Penalizes missing hard-to-find legal entities like PENALTY or RATE
class LegalFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.smoothing = smoothing  # Add smoothing to handle inconsistent labels

    def forward(self, inputs, targets):
        # Apply label smoothing manually for better noise handling
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            ignore_mask = targets == -100
            # Fill valid targets
            targets_safe = targets.clone()
            targets_safe[ignore_mask] = 0
            true_dist.scatter_(1, targets_safe.unsqueeze(1), 1.0 - self.smoothing)

        ce_loss = -(true_dist * log_probs).sum(dim=-1)
        # Apply Focal weighting
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss[~ignore_mask].mean()


class RobustNERModel(nn.Module):
    """Wrapper to add Multi-Sample Dropout for noise resistance."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        # Standard PhoBERT hidden size is 768
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Multi-sample dropout: average the logits from 5 different masks
        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(sequence_output))
        logits /= len(self.dropouts)

        return {"logits": logits}


class StableTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Handle the wrapper or raw model
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = LegalFocalLoss(smoothing=0.1)
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

    raw_model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    # Apply the Robust Wrapper
    model = RobustNERModel(raw_model)

    training_args = TrainingArguments(
        output_dir="./models/robust_ner_checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,  # Lower LR for deeper fine-tuning
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.05,  # AGGRESSIVE decay to ignore inconsistent noise
        warmup_steps=300,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = StableTrainer(
        model=model,  # Trainer will use our wrapped model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
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
