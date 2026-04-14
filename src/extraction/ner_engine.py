import os

import torch
from transformers import AutoTokenizer, pipeline
from underthesea import word_tokenize

MODEL_PATH = "models/ultra_ner"
_ner_pipeline = None


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        if os.path.exists(MODEL_PATH):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
            _ner_pipeline = pipeline(
                "token-classification",
                model=MODEL_PATH,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float32,  # Stability for inference
            )
    return _ner_pipeline


def extract_ultra_entities(text: str) -> list[dict]:
    """Unified model for NER. Pre-segments text for PhoBERT."""
    if not text or not text.strip():
        return []

    pipe = get_ner_pipeline()
    if not pipe:
        return []

    # 1. Segment text
    segmented_text = word_tokenize(text, format="text")
    results = pipe(segmented_text)

    # 2. To fix the span shift, we map offsets back to the raw text
    entities = []
    raw_text_lower = text.lower()
    search_idx = 0

    for res in results:
        # PhoBERT artifact cleaning
        clean_word = res["word"].replace("_", " ").strip()
        if not clean_word:
            continue

        # Find where this predicted word actually exists in the original text
        actual_start = raw_text_lower.find(clean_word.lower(), search_idx)

        if actual_start != -1:
            actual_end = actual_start + len(clean_word)
            search_idx = actual_end  # Move pointer forward

            entities.append(
                {
                    "text": text[actual_start:actual_end],  # Use original casing
                    "label": res["entity_group"],
                    "span": (actual_start, actual_end),
                }
            )
    return entities


def extract_entities(text: str) -> list[dict]:
    """
    [Task 2.1] Named Entity Recognition.
    Filters out technical labels (OBJECT, PREDICATE) and maps ULTRA-NER
    to standard legal entities.
    """
    raw = extract_ultra_entities(text)
    # Professor only wants standard entities for Task 2.1
    legal_entities = []
    for e in raw:
        if e["label"] in ["PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"]:
            legal_entities.append(e)
    return legal_entities


def extract_for_srl_and_chunking(text: str) -> list[dict]:
    """Returns all labels including OBJECT and PREDICATE for downstream DL tasks."""
    return extract_ultra_entities(text)
