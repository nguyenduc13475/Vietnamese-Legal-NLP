import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    RobertaModel,
    RobertaPreTrainedModel,
    Trainer,
    TrainingArguments,
)


# Fixed Inheritance: Use RobertaPreTrainedModel which PhoBERT is based on
class LegalPhoBERTNER(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # BiLSTM for high-level sequence modeling
        self.bilstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        # Enhanced Head: LayerNorm + GELU (more stable than ReLU for transformers)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        self.loss_weights = weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Apply BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)

        # Normalize and classify
        lstm_output = self.norm(lstm_output)
        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            # Flatten context for standard NER loss calculation
            loss_fct = nn.CrossEntropyLoss(weight=self.loss_weights)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels),
            )
            loss = loss_fct(active_logits, active_labels)

        return {"loss": loss, "logits": logits}


def calculate_label_weights(dataset, num_labels):
    """Calculate inverse frequency weights to handle class imbalance."""
    counts = Counter()
    for item in dataset["ner_tags"]:
        # Map string tags to IDs if needed, but here they are IDs
        counts.update(item)

    weights = torch.ones(num_labels)
    total = sum(counts.values())
    for i in range(num_labels):
        if counts[i] > 0:
            # Formula: total / (num_classes * count)
            weights[i] = total / (num_labels * counts[i])

    # Normalize weights to avoid exploding gradients
    weights = weights / weights.mean()
    return weights


def load_custom_data(file_path):
    """
    Loading NER training data from JSON.
    NER tag format: ["O", "B-PARTY", "I-PARTY", "B-OBJECT", ...]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data not found at {file_path}. Please generate data before training."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def train(model_name: str, epochs: int, batch_size: int, learning_rate: float):
    print(f"Initializing Enhanced Training Pipeline for {model_name}...")
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

    train_dataset_raw = load_custom_data("data/annotated/ner_train.json")
    eval_dataset_raw = load_custom_data("data/annotated/ner_test.json")

    # Calculate class weights for imbalance handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_weights = calculate_label_weights(train_dataset_raw, len(id2label)).to(device)

    # Load custom config and model
    config = AutoConfig.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    # Initialize from pretrained PhoBERT weights rather than random config
    model = LegalPhoBERTNER.from_pretrained(
        model_name, config=config, weights=label_weights
    )

    # Freeze the first 6 layers of BERT to focus training on the LSTM head and higher semantic features
    # This often helps significantly with small, domain-specific datasets
    for name, param in model.roberta.named_parameters():
        if any(f"layer.{i}." in name for i in range(6)):
            param.requires_grad = False
    print("Frozen first 6 layers of PhoBERT to stabilize training.")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for i, ids in enumerate(examples["input_ids"]):
            tag_list = examples["ner_tags"][i]
            # Convert string labels to IDs if they are strings
            tag_ids = [label2id[t] if isinstance(t, str) else t for t in tag_list]

            input_ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
            label_ids = [-100] + tag_ids + [-100]

            if len(input_ids) > 256:
                input_ids = input_ids[:255] + [tokenizer.eos_token_id]
                label_ids = label_ids[:255] + [-100]

            tokenized_inputs["input_ids"].append(input_ids)
            tokenized_inputs["attention_mask"].append([1] * len(input_ids))
            tokenized_inputs["labels"].append(label_ids)
        return tokenized_inputs

    train_dataset = train_dataset_raw.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset_raw.column_names,
    )
    eval_dataset = eval_dataset_raw.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=eval_dataset_raw.column_names,
    )

    training_args = TrainingArguments(
        output_dir="./models/ultra_ner",
        eval_strategy="epoch",
        learning_rate=3e-5,  # Lower LR for stability with custom heads
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Cosine annealing helps reach higher F1 peaks
        warmup_ratio=0.1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        fp16=False,  # CRITICAL: Disable FP16 to fix NaN Grad Norm
        max_grad_norm=1.0,  # CRITICAL: Clip gradients to prevent explosion
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("Starting Optimized Deep NER Training...")
    trainer.train()

    # Save both model and tokenizer
    output_dir = "./models/fine_tuned_ner"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Success! Model with BiLSTM-head saved at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ULTRA-NER Model for Legal Contracts"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vinai/phobert-base",
        help="Pretrained model (e.g., vinai/phobert-base)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
