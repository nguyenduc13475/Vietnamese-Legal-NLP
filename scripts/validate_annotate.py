import json
import os

VALID_INTENTS = {"Obligation", "Prohibition", "Right", "Termination Condition", "Other"}
VALID_ENTITIES = {"PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"}


def validate_annotated_data(file_path="data/annotated_raw.json"):
    print(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        print("Error: File not found. Please check the path again.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(
                f"Error: JSON file has syntax errors (missing brackets, extra commas, etc.). Details: {e}"
            )
            return

    print(f"Checking the {len(data)} clause...\n")

    total_error_clauses = 0

    for index, item in enumerate(data):
        clause = item.get("clause", "")
        intent = item.get("intent", "")
        entities = item.get("entities", [])

        clause_errors = []

        if intent not in VALID_INTENTS:
            clause_errors.append(f"Invalid Intent: '{intent}'")

        current_idx = 0

        for ent in entities:
            text = ent.get("text", "")
            label = ent.get("label", "")

            if label not in VALID_ENTITIES:
                clause_errors.append(
                    f"Invalid label: '{label}' (belongs to text: '{text}')"
                )

            found_idx = clause.find(text, current_idx)

            if found_idx == -1:
                if text in clause:
                    clause_errors.append(
                        f"Text is out of order, redundant, OR illegally nested: '{text}'"
                    )
                else:
                    clause_errors.append(f"Fabricated/non-matching text: '{text}'")
            else:
                current_idx = found_idx + len(text)

        if clause_errors:
            total_error_clauses += 1
            print(f"Error detected at Index [{index}]:")
            print(f"Clause: {clause[:80]}...")
            for err in clause_errors:
                print(f"   ❌ {err}")
            print("-" * 50)

    print("\n" + "=" * 50)
    if total_error_clauses == 0:
        print(
            "PERFECT! Your data file is completely valid (Standard 1:1, Left to Right, NO nesting)."
        )
    else:
        print(f"SUMMARY: There are {total_error_clauses} clauses that are faulty.")


if __name__ == "__main__":
    validate_annotated_data("data/annotated_raw.json")
