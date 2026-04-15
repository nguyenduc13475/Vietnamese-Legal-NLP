import os

import torch
from transformers import AutoTokenizer
from underthesea import word_tokenize

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

    # Manual Offset Calculation for Python Tokenizer
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    offsets = []
    cursor = 0
    for token in tokens:
        if token in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
            offsets.append((0, 0))
            continue

        # Clean token for alignment: remove subword markers (@@) and space markers (_)
        clean_tok = token.replace("@@", "").replace("_", " ").strip()
        if not clean_tok:
            offsets.append((cursor, cursor))
            continue

        start_idx = text.find(clean_tok, cursor)
        if start_idx == -1:  # Case-insensitive fallback
            start_idx = text.lower().find(clean_tok.lower(), cursor)

        if start_idx != -1:
            end_idx = start_idx + len(clean_tok)
            offsets.append((start_idx, end_idx))
            cursor = end_idx
        else:
            offsets.append((cursor, cursor))

    # ID to Label Mapping
    id2label = {
        0: "O",
        1: "PARTY",
        3: "MONEY",
        5: "DATE",
        7: "RATE",
        9: "PENALTY",
        11: "LAW",
        13: "OBJECT",
        15: "PREDICATE",
    }

    entities = []
    for i, p_id in enumerate(preds):
        label = id2label.get(
            p_id if p_id % 2 != 0 else p_id - 1 if p_id > 0 else 0, "O"
        )
        if label == "O":
            continue

        start, end = offsets[i]
        if start == end:
            continue

        entities.append(
            {"text": text[start:end], "label": label, "span": (int(start), int(end))}
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
