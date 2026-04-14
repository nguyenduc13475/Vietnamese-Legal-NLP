import json
import os
from collections import Counter

# Updated validation sets
VALID_INTENTS = {"Obligation", "Prohibition", "Right", "Termination Condition", "Other"}
VALID_NER_LABELS = {"PARTY", "DATE", "LAW", "PENALTY", "MONEY", "RATE", "OBJECT"}
VALID_SRL_LABELS = {
    "AGENT",
    "RECIPIENT",
    "THEME",
    "NAME",
    "TIME",
    "CONDITION",
    "TRAIT",
    "LOCATION",
    "METHOD",
    "ABOUT",
    "PREDICATE",
}


def validate_annotated_data(file_path="data/annotated_raw.json"):
    print(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: JSON syntax error. Details: {e}")
            return

    print(f"Checking {len(data)} clauses...\n")
    total_error_clauses = 0

    for index, item in enumerate(data):
        clause = item.get("clause", "")
        intent = item.get("intent", "")
        ultra_ner = item.get("ultra_ner", [])
        srl_roles = item.get("srl_roles", [])

        clause_errors = []

        # 1. Validate Intent
        if intent not in VALID_INTENTS:
            clause_errors.append(f"Invalid Intent: '{intent}'")

        # 2. Validate ultra_ner (Rule 1: Left to Right & Rule 2: No Nesting)
        current_ner_idx = 0
        for ent in ultra_ner:
            text = ent.get("text", "")
            label = ent.get("label", "")

            if label not in VALID_NER_LABELS:
                clause_errors.append(f"Invalid NER label: '{label}' in text '{text}'")

            found_idx = clause.find(text, current_ner_idx)
            if found_idx == -1:
                if text in clause:
                    clause_errors.append(
                        f"NER Text out of order or overlapping: '{text}'"
                    )
                else:
                    clause_errors.append(
                        f"NER Fabricated text (not in clause): '{text}'"
                    )
            else:
                current_ner_idx = found_idx + len(text)

        # 3. Validate srl_roles (Rule 1: Left to Right & Uniqueness Rule)
        current_srl_idx = 0
        role_counts = Counter()  # Tracks occurrences of each label

        for role in srl_roles:
            text = role.get("text", "")
            label = role.get("label", "")

            # Increment count for this specific label
            role_counts[label] += 1

            if label not in VALID_SRL_LABELS:
                clause_errors.append(f"Invalid SRL label: '{label}' in text '{text}'")

            # Check position for SRL
            found_idx = clause.find(text, current_srl_idx)
            if found_idx == -1:
                if text not in clause:
                    clause_errors.append(f"SRL Fabricated text: '{text}'")
            else:
                current_srl_idx = found_idx + len(text)

        # 4. Check for ANY duplicate roles
        for label, count in role_counts.items():
            if count > 1:
                clause_errors.append(
                    f"Duplicate Role Detected: '{label}' occurs {count} times"
                )

        # Output errors
        if clause_errors:
            total_error_clauses += 1
            print(f"Error detected at Index [{index}]:")
            print(f"Clause: {clause[:100]}...")
            for err in clause_errors:
                print(f"   ❌ {err}")
            print("-" * 50)

    print("\n" + "=" * 50)
    if total_error_clauses == 0:
        print("PERFECT! All ultra_ner and srl_roles follow the guidelines.")
    else:
        print(f"SUMMARY: There are {total_error_clauses} faulty clauses.")


if __name__ == "__main__":
    validate_annotated_data("data/annotated_raw.json")
