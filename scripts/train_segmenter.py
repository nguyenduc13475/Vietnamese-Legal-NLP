import json

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def train_segmenter():
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load auto-annotated data (Labels 0-5)
    with open("data/annotated/segment_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    dataset = Dataset.from_list(train_data)

    def align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=256,
        )
        labels = []
        for i, label in enumerate(examples["segment_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    tokenized_ds = dataset.map(align_labels, batched=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=6)

    args = TrainingArguments(
        output_dir="models/fine_tuned_segmenter",
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    print("Training DL Segmenter (Type 1 & 2)...")
    trainer.train()
    trainer.save_model("models/fine_tuned_segmenter")


if __name__ == "__main__":
    train_segmenter()
