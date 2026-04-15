import argparse
import json
import os

from src.extraction.intent_classifier import classify_intent
from src.extraction.ner_engine import extract_entities, extract_ultra_entities
from src.extraction.srl_engine import extract_srl
from src.preprocessing.chunker import chunk_np
from src.preprocessing.parser import parse_dependency
from src.preprocessing.segmenter import segment_clauses


def main():
    parser = argparse.ArgumentParser(description="Run NLP Pipeline for Legal Contracts")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/hop_dong_thue_nha.txt",
        help="Path to input text file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print("1.1 Clause Splitting...")
    # Handle both dict-return and string-list return from segmenter
    raw_clauses = segment_clauses(text)
    clauses = [c["text"] if isinstance(c, dict) else c for c in raw_clauses]

    with open(os.path.join(args.output_dir, "clauses.txt"), "w", encoding="utf-8") as f:
        for clause in clauses:
            f.write(clause + "\n")

    print("1.2 Noun Phrase Clustering...")
    chunks_results = []
    for clause in clauses:
        chunks_results.append(chunk_np(clause))
    with open(os.path.join(args.output_dir, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk_list in chunks_results:
            for word, tag in chunk_list:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")

    print("1.3 Dependency Analysis...")
    dep_results = []
    for clause in clauses:
        dep_results.append({"clause": clause, "dependencies": parse_dependency(clause)})
    with open(
        os.path.join(args.output_dir, "dependency.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dep_results, f, ensure_ascii=False, indent=4)

    print("2.1 Named Entity Recognition (NER)...")
    ner_results = []
    ultra_features = []  # For SRL
    for clause in clauses:
        ents = extract_entities(clause)  # Task 2.1 Output
        ner_results.append({"clause": clause, "entities": ents})
        ultra_features.append(extract_ultra_entities(clause))  # Features for 2.2

    print("2.2 Semantic Role Labeling (SRL)...")
    srl_results = []
    for i, clause in enumerate(clauses):
        entities = ultra_features[i]  # Use full features (Predicate/Object included)
        srl_results.append({"clause": clause, "srl": extract_srl(clause, entities)})
    with open(
        os.path.join(args.output_dir, "srl_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(srl_results, f, ensure_ascii=False, indent=4)

    print("2.3 Intent Classification...")
    with open(
        os.path.join(args.output_dir, "intent_classification.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for clause in clauses:
            intent = classify_intent(clause)
            f.write(f"{clause}\t{intent}\n")

    print(
        f"Completed! Results exported in Assignment format to directory: {args.output_dir}/"
    )


if __name__ == "__main__":
    main()
