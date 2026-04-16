import argparse
import json
import os

# Internal module imports
from src.extraction.intent_classifier import classify_intent
from src.extraction.ner_engine import extract_entities, extract_ultra_entities
from src.extraction.srl_engine import extract_srl
from src.preprocessing.chunker import chunk_np
from src.preprocessing.parser import parse_dependency
from src.preprocessing.segmenter import segment_clauses


def setup_directories(input_dir, output_dir):
    """Ensure input and output directories exist."""
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(input_dir, "raw_contracts.txt")
    if not os.path.exists(raw_path):
        print(f"[*] Creating sample input file at {raw_path}")
        sample_text = (
            "Bên B sẽ thanh toán toàn bộ tiền thuê trước ngày 5 hàng tháng, và "
            "nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng.\n"
            "Bên A phải bàn giao vật tư cho bên B trước ngày 15/05/2026."
        )
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(sample_text)


def main():
    parser = argparse.ArgumentParser(
        description="Legal Contract NLP Pipeline CLI (Assignment Format)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="input/raw_contracts.txt",
        help="Path to raw contract text",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for assignment outputs",
    )
    args = parser.parse_args()

    input_dir = os.path.dirname(args.input)
    setup_directories(input_dir, args.output_dir)

    if not os.path.exists(args.input):
        print(f"[!] Error: Input file {args.input} not found.")
        return

    # 1. Read Raw Text
    with open(args.input, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[*] Starting Pipeline for: {args.input}")

    # --- TASK 1.1: Clause Segmentation ---
    print("[1/6] Task 1.1: Segmenting clauses...")
    clause_data = segment_clauses(raw_text)
    # Extract only text from dict objects returned by segmenter
    clauses = [c["text"] if isinstance(c, dict) else c for c in clause_data]

    with open(os.path.join(args.output_dir, "clauses.txt"), "w", encoding="utf-8") as f:
        for clause in clauses:
            f.write(clause + "\n")

    # Containers for results
    all_chunks = []
    all_dependencies = []
    all_ner = []
    all_srl = []
    all_intents = []

    # Process each clause through the rest of the tasks
    total = len(clauses)
    for i, clause in enumerate(clauses, 1):
        print(f"    -> Processing clause {i}/{total}...", end="\r")

        # --- TASK 1.2: NP Chunking ---
        chunks = chunk_np(clause)
        all_chunks.append(chunks)

        # --- TASK 1.3: Dependency Parsing ---
        deps = parse_dependency(clause)
        all_dependencies.append({"clause": clause, "dependencies": deps})

        # --- TASK 2.1: Custom NER ---
        # Task 2.1 requires filtered legal entities
        ents = extract_entities(clause)
        all_ner.append({"clause": clause, "entities": ents})

        # --- TASK 2.2: Semantic Role Labeling (SRL) ---
        # Note: SRL requires ultra-entities (including Predicate/Object) to function
        ultra_ents = extract_ultra_entities(clause)
        srl_res = extract_srl(clause, ultra_ents)
        all_srl.append({"clause": clause, "srl": srl_res})

        # --- TASK 2.3: Intent Classification ---
        intent = classify_intent(clause)
        all_intents.append((clause, intent))

    print("\n[2/6] Task 1.2: Exporting chunks.txt...")
    with open(os.path.join(args.output_dir, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk_list in all_chunks:
            for word, tag in chunk_list:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")  # Blank line between clauses per standard NLP format

    print("[3/6] Task 1.3: Exporting dependency.json...")
    with open(
        os.path.join(args.output_dir, "dependency.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_dependencies, f, ensure_ascii=False, indent=4)

    print("[4/6] Task 2.1: Exporting ner_results.json...")
    with open(
        os.path.join(args.output_dir, "ner_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_ner, f, ensure_ascii=False, indent=4)

    print("[5/6] Task 2.2: Exporting srl_results.json...")
    with open(
        os.path.join(args.output_dir, "srl_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_srl, f, ensure_ascii=False, indent=4)

    print("[6/6] Task 2.3: Exporting intent_classification.txt...")
    with open(
        os.path.join(args.output_dir, "intent_classification.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for clause, intent in all_intents:
            f.write(f"{clause}\t{intent}\n")

    print("\n[+] Pipeline Completed Successfully!")
    print(f"[+] All results exported to the '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()
