import argparse
import glob
import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.intent_classifier import classify_intent
from src.extraction.ner_engine import extract_entities, extract_ultra_entities
from src.extraction.srl_engine import extract_srl
from src.preprocessing.chunker import chunk_np
from src.preprocessing.parser import parse_dependency
from src.preprocessing.segmenter import segment_clauses
from src.qa.retriever import LegalRetriever

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
        raw_clauses_dicts = segment_clauses(text)

        # Extract the human-readable title from the batch
        doc_title = os.path.basename(file_path)  # Default to filename
        for item in raw_clauses_dicts:
            if item.get("is_title"):
                doc_title = item["text"]
                break

        valid_texts = []
        metadata = []

        # Filter out invalid clauses and unpack the dictionary
        for item in raw_clauses_dicts:
            if len(item["text"].strip()) > 10:
                valid_texts.append(item["text"])
                # Initialize metadata with context, title, and flags
                metadata.append(
                    {
                        "contract_title": doc_title,
                        "context": item["context"],
                        "is_title": str(item.get("is_title", False)),
                        "aliases": str(item.get("aliases", "[]")),
                    }
                )

        if not valid_texts:
            print("  -> Warning: No valid clauses found. Skipping.")
            continue

        total_c = len(valid_texts)

        print(
            "  -> Extracting features sequentially (avoiding Python GIL and CUDA thread locks)..."
        )

        for c_idx, text in enumerate(valid_texts):
            # Task 2.1: Get filtered entities for metadata storage
            ents_task_2_1 = extract_entities(text)

            ultra_ents = extract_ultra_entities(text)

            deps = parse_dependency(text)
            chunks = chunk_np(text)

            # IMPORTANT: SRL needs ultra_ents to function correctly
            srl = extract_srl(text, ultra_ents, deps, chunks)
            intent = classify_intent(text)

            metadata[c_idx].update(
                {
                    "source": os.path.basename(file_path),
                    "intent": intent,
                    "np_chunks": str(chunks),
                    "entities": str(
                        [
                            {"text": e["text"], "label": e["label"]}
                            for e in ents_task_2_1
                        ]
                    ),
                    "predicate": str(srl.get("predicate", "")),
                    "srl_roles": str(srl.get("roles", {})),
                    "dependencies": str(
                        [
                            {
                                "id": d["id"],
                                "token": d["token"],
                                "relation": d["relation"],
                                "head_index": d["head_index"],
                                "head_token": d.get("head_token", ""),
                            }
                            for d in deps
                        ]
                    ),
                }
            )

            if (c_idx + 1) % 10 == 0 or (c_idx + 1) == total_c:
                print(f"  -> Processed {c_idx + 1}/{total_c} clauses...", end="\r")

        print(f"\n  -> Inserting {total_c} clauses into Vector DB...")
        retriever.add_clauses(valid_texts, metadata)
        total_clauses += len(valid_texts)

    print(f"Success! Indexed total {total_clauses} clauses into Vector DB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest contract text into Vector DB")
    parser.add_argument(
        "--input", required=True, help="Path to raw contract txt or directory"
    )
    args = parser.parse_args()
    build_db(args.input)
