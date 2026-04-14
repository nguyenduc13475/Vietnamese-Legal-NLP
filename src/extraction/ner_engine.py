import os

import torch
from transformers import AutoTokenizer
from underthesea import word_tokenize

MODEL_PATH = "models/fine_tuned_ner"
_ner_model = None
_ner_tokenizer = None


def get_ner_resources():
    """Load the custom architecture model without using the generic pipeline."""
    global _ner_model, _ner_tokenizer
    if _ner_model is None:
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
            from scripts.train_ner import LegalPhoBERTNER

            _ner_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            # Use AutoConfig to map the custom model back
            _ner_model = LegalPhoBERTNER.from_pretrained(MODEL_PATH)
            _ner_model.eval()
            if torch.cuda.is_available():
                _ner_model.to("cuda")
    return _ner_model, _ner_tokenizer


def extract_ultra_entities(text: str) -> list[dict]:
    """Unified model for NER using Enhanced BiLSTM-head model."""
    if not text or not text.strip():
        return []

    model, tokenizer = get_ner_resources()
    if not model:
        return []

    segmented_text = word_tokenize(text, format="text")
    inputs = tokenizer(
        segmented_text, return_tensors="pt", truncation=True, max_length=256
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

    # Convert predictions back to labels and map spans
    id2label = model.config.id2label
    # Standard logic to aggregate tokens into entity groups...
    # (Simplified for the response, using a character-match approach as in original)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    current_pos = 0
    raw_text_lower = text.lower()

    # Simple aggregation logic for custom model
    for i, pred_id in enumerate(predictions):
        label = id2label[pred_id]
        if label == "O" or tokens[i] in [
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.pad_token,
        ]:
            continue

        clean_tok = tokens[i].replace("@@", "").replace("_", " ").strip()
        if not clean_tok:
            continue

        start_idx = raw_text_lower.find(clean_tok.lower(), current_pos)
        if start_idx != -1:
            end_idx = start_idx + len(clean_tok)
            entities.append(
                {
                    "text": text[start_idx:end_idx],
                    "label": label.replace("B-", "").replace("I-", ""),
                    "span": (start_idx, end_idx),
                }
            )
            current_pos = end_idx

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
