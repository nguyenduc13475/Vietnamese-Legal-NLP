import argparse
import glob
import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.intent_classifier import classify_intent
from src.extraction.ner_engine import extract_entities
from src.extraction.srl_engine import extract_srl
from src.preprocessing.chunker import chunk_np
from src.preprocessing.parser import parse_dependency
from src.preprocessing.segmenter import segment_clauses
from src.qa.retriever import LegalRetriever

# Suppress lingering tokenization FutureWarnings from Langchain's internal initializations
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)


def build_db(input_path: str):
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.txt"))
    else:
        files = [input_path]

    retriever = LegalRetriever()
    total_clauses = 0

    print(f"Found {len(files)} contract files to process.")

    for file_idx, file_path in enumerate(files, 1):
        print(
            f"[{file_idx}/{len(files)}] Reading and processing: {os.path.basename(file_path)}"
        )
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Segment and filter out extremely short/meaningless clauses
        raw_clauses = segment_clauses(text)
        clauses = [c for c in raw_clauses if len(c.strip()) > 10]

        if not clauses:
            print("  -> Warning: No valid clauses found. Skipping.")
            continue

        metadata = []

        total_c = len(clauses)
        for c_idx, clause in enumerate(clauses, 1):
            if c_idx % 10 == 0 or c_idx == total_c:
                print(
                    f"  -> Extracting features: {c_idx}/{total_c} clauses...", end="\r"
                )

            ents = extract_entities(clause)
            deps = parse_dependency(clause)
            chunks = chunk_np(clause)
            srl = extract_srl(clause, ents, deps, chunks)
            intent = classify_intent(clause)

            meta = {
                "source": os.path.basename(file_path),
                "intent": intent,
                "entities": str([e["text"] for e in ents]),
                "predicate": str(srl.get("predicate", "")),
            }
            metadata.append(meta)

        print(f"\n  -> Inserting {total_c} clauses into Vector DB...")
        retriever.add_clauses(clauses, metadata)
        total_clauses += len(clauses)

    print(f"Success! Indexed total {total_clauses} clauses into Vector DB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest contract text into Vector DB")
    parser.add_argument(
        "--input", required=True, help="Path to raw contract txt or directory"
    )
    args = parser.parse_args()
    build_db(args.input)
