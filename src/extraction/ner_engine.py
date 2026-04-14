import os

import torch
from transformers import AutoTokenizer, pipeline

MODEL_PATH = "models/ultra_ner"
_ner_pipeline = None


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
            tokenizer = AutoTokenizer.from_pretrained(
                "Fsoft-AIC/videberta-xsmall", use_fast=True
            )
            _ner_pipeline = pipeline(
                "token-classification",
                model=MODEL_PATH,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
            )
    return _ner_pipeline


def extract_ultra_entities(text: str) -> list[dict]:
    """Unified model for NER, NP Chunking (OBJECT), and SRL features (PREDICATE)."""
    if not text or not text.strip():
        return []

    pipe = get_ner_pipeline()
    if not pipe:
        return []

    results = pipe(text)
    entities = []
    for res in results:
        entities.append(
            {
                "text": res["word"].replace(" ", " ").strip(),
                "label": res["entity_group"],
                "span": (res["start"], res["end"]),
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
