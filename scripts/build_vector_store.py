import argparse
import concurrent.futures
import glob
import os
import sys
import warnings

import torch

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
        raw_clauses_dicts = segment_clauses(text)

        valid_texts = []
        metadata = []

        # Filter out invalid clauses and unpack the dictionary
        for item in raw_clauses_dicts:
            if len(item["text"].strip()) > 10:
                valid_texts.append(item["text"])
                # Initialize metadata with context and the is_title flag
                metadata.append(
                    {
                        "context": item["context"],
                        "is_title": str(item.get("is_title", False)),
                    }
                )

        if not valid_texts:
            print("  -> Warning: No valid clauses found. Skipping.")
            continue

        total_c = len(valid_texts)

        # Worker function to run independent NLP engines in parallel
        def process_clause(text, idx):
            ents = extract_entities(text)
            deps = parse_dependency(text)
            chunks = chunk_np(text)
            srl = extract_srl(text, ents, deps, chunks)
            intent = classify_intent(text)
            return idx, ents, deps, chunks, srl, intent

        # 8 threads is the sweet spot for a Colab T4 to max out CUDA
        # without running out of VRAM (OOM). Use 4 if strictly on CPU.
        workers = 8 if torch.cuda.is_available() else 4
        print(f"  -> Extracting features concurrently using {workers} threads...")

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all clauses to the thread pool
            futures = [
                executor.submit(process_clause, text, i)
                for i, text in enumerate(valid_texts)
            ]

            # As threads finish, collect results and map them back to the correct metadata index
            for future in concurrent.futures.as_completed(futures):
                c_idx, ents, deps, chunks, srl, intent = future.result()

                metadata[c_idx].update(
                    {
                        "source": os.path.basename(file_path),
                        "intent": intent,
                        "np_chunks": str(chunks),
                        "entities": str(
                            [{"text": e["text"], "label": e["label"]} for e in ents]
                        ),
                        "predicate": str(srl.get("predicate", "")),
                        "srl_roles": str(srl.get("roles", {})),
                        "dependencies": str(
                            [f"{d['token']}({d['relation']})" for d in deps]
                        ),
                    }
                )

                completed += 1
                if completed % 10 == 0 or completed == total_c:
                    print(f"  -> Processed {completed}/{total_c} clauses...", end="\r")

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
