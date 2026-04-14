import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer

from src.extraction.srl_engine import ID2SRL, JointSRLModel

# Reverse mapping for training
SRL2ID = {v: k for k, v in ID2SRL.items()}


class LegalSRLDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data[item]
        tokens = row["tokens"]
        srl_tags = row["srl_tags"]

        # In a real training scenario, we would pre-calculate NER and DEP IDs
        # based on the gold labels or previous stage outputs.
        # For this script, we assume they are provided in the annotated data or simulated.
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = []
        word_ids = encoding.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(SRL2ID.get(srl_tags[word_idx], 0))

        # Mocking O2 features for training demonstration -
        # in production, these are piped from Stanza + ULTRA-NER
        ner_ids = torch.zeros(self.max_len, dtype=torch.long)
        dep_ids = torch.zeros(self.max_len, dtype=torch.long)
        p_ner_ids = torch.zeros(self.max_len, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels),
            "ner_ids": ner_ids,
            "dep_ids": dep_ids,
            "p_ner_ids": p_ner_ids,
        }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Fsoft-AIC/videberta-xsmall")
    model = JointSRLModel().to(device)

    train_ds = LegalSRLDataset("data/annotated/srl_train.json", tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print("Starting Heterogeneous SRL Training...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["ner_ids"].to(device),
                batch["dep_ids"].to(device),
                batch["p_ner_ids"].to(device),
            )

            loss = criterion(logits.view(-1, 11), batch["labels"].to(device).view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs("models/ultra_srl", exist_ok=True)
    torch.save(model.state_dict(), "models/ultra_srl/pytorch_model.bin")
    tokenizer.save_pretrained("models/ultra_srl")
    print("SRL Model saved to models/ultra_srl/")


if __name__ == "__main__":
    train()
