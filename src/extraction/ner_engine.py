import os

import torch
from transformers import AutoTokenizer
from underthesea import word_tokenize

from src.utils.constants import NER_LABEL_MAP
from src.utils.model_loader import load_robust_classification_model

MODEL_PATH = "models/ner"
_ner_model = None
_ner_tokenizer = None


def get_ner_resources():
    global _ner_model, _ner_tokenizer
    if _ner_model is None:
        if os.path.exists(MODEL_PATH):
            # Load tokenizer from MODEL_PATH to use local vocab.txt
            _ner_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            _ner_model = load_robust_classification_model(
                MODEL_PATH, num_labels=17, is_token_level=True
            )
    return _ner_model, _ner_tokenizer


def extract_ultra_entities(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
    model, tokenizer = get_ner_resources()
    if not model:
        return []

    device = next(model.parameters()).device
    segmented_text = word_tokenize(text, format="text")
    inputs = tokenizer(
        segmented_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs)["logits"]
        preds = torch.argmax(logits, dim=2)[0].cpu().numpy()

    # Manual Sequential Offset Calculation for PhoBERT (No Fast Tokenizer support)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    offsets = []
    cursor = 0
    # Create a space-agnostic version for easier index matching
    text_clean = text.lower()

    for token in tokens:
        if token in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
            offsets.append((0, 0))
            continue

        # PhoBERT uses '_' for spaces and '@@' for subwords
        clean_tok = token.replace("@@", "").replace("_", " ").strip().lower()

        if not clean_tok:
            offsets.append((cursor, cursor))
            continue

        # Find the next occurrence starting from current cursor
        start_idx = text_clean.find(clean_tok, cursor)

        if start_idx != -1:
            end_idx = start_idx + len(clean_tok)
            offsets.append((start_idx, end_idx))
            cursor = end_idx
        else:
            # Fallback for complex word segmentation mismatches
            offsets.append((cursor, cursor))

    # Invert the map for prediction decoding
    ID2LABEL = {v: k for k, v in NER_LABEL_MAP.items()}
    entities = []
    current_entity = None

    for i, p_id in enumerate(preds):
        raw_label = ID2LABEL.get(p_id, "O")

        if raw_label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        start, end = offsets[i]
        if start == end:
            continue

        # Clean the label for the final output
        label = raw_label.replace("B-", "").replace("I-", "")

        if raw_label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": text[start:end],
                "label": label,
                "span": (int(start), int(end)),
            }
        elif raw_label.startswith("I-"):
            if current_entity and current_entity["label"] == label:
                # Merge with previous token
                curr_start = current_entity["span"][0]
                current_entity["span"] = (curr_start, int(end))
                current_entity["text"] = text[curr_start : int(end)]
            else:
                # Treat as B- if we see an I- without a matching preceding B-
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": text[start:end],
                    "label": label,
                    "span": (int(start), int(end)),
                }

    if current_entity:
        entities.append(current_entity)

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
