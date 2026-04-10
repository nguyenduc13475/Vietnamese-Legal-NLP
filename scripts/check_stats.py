import json
import os
from collections import Counter

NER_B_TAGS = {1: "PARTY", 3: "MONEY", 5: "DATE", 7: "RATE", 9: "PENALTY", 11: "LAW"}


def count_intent_labels(filepath):
    if not os.path.exists(filepath):
        return {"Error": f"File {filepath} not found"}

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [item.get("label", "Unknown") for item in data]
    return dict(Counter(labels))


def count_ner_entities(filepath):
    if not os.path.exists(filepath):
        return {"Lỗi": f"Không tìm thấy {filepath}"}

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    entity_counts = Counter()
    for item in data:
        tags = item.get("ner_tags", [])
        for tag in tags:
            if tag in NER_B_TAGS:
                entity_counts[NER_B_TAGS[tag]] += 1

    return dict(entity_counts)


def print_stats(title, intent_path, ner_path):
    print("=" * 50)
    print(f"Data set stats: {title.upper()}")
    print("=" * 50)

    # Thống kê Intent
    print("\n1. Intent Classification - No. Clauses:")
    intent_stats = count_intent_labels(intent_path)
    for k, v in sorted(intent_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {k:<25}: {v}")

    # Thống kê NER
    print("\n2. NER - No. Entities:")
    ner_stats = count_ner_entities(ner_path)
    for k, v in sorted(ner_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {k:<25}: {v}")
    print("\n")


if __name__ == "__main__":
    print_stats(
        "TRAIN", "data/annotated/intent_train.json", "data/annotated/ner_train.json"
    )

    print_stats(
        "TEST", "data/annotated/intent_test.json", "data/annotated/ner_test.json"
    )
