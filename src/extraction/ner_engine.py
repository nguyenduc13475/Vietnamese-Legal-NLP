import os
import re

import torch
from transformers import AutoTokenizer, pipeline

MODEL_PATH = "models/ultra_ner"  # Fine-tuned ViDeBERTa-xsmall
_ner_pipeline = None


def get_ner_pipeline():
    """Lazy load the pipeline to avoid reloading on every function call."""
    global _ner_pipeline
    if _ner_pipeline is None:
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
            try:
                device_id = 0 if torch.cuda.is_available() else -1
                tokenizer = AutoTokenizer.from_pretrained(
                    "Fsoft-AIC/videberta-xsmall",
                    clean_up_tokenization_spaces=True,
                    model_max_length=256,
                )
                _ner_pipeline = pipeline(
                    "token-classification",
                    model=MODEL_PATH,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    ignore_labels=["O"],
                    device=device_id,
                )
            except Exception as e:
                print(
                    f"Warning: Could not load NER model from {MODEL_PATH}. Error: {e}"
                )
                _ner_pipeline = "fallback"
        else:
            _ner_pipeline = "fallback"
    return _ner_pipeline


def extract_ultra_entities(text: str) -> list[dict]:
    """
    ULTRA-NER: Unified model for NER + NP-Chunking + Predicate detection.
    Labels: PARTY, MONEY, DATE, RATE, PENALTY, LAW, OBJECT, PREDICATE
    Returns the standard interface: {"text": str, "label": str, "span": tuple}
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    entities = []
    ner_pipe = get_ner_pipeline()

    if ner_pipe and ner_pipe != "fallback":
        # 1. Machine Learning Inference
        ml_entities = ner_pipe(text)

        for ent in ml_entities:
            label = ent.get("entity_group", "")
            if label:
                # Clean up ViDeBERTa special token markers (U+2581)
                clean_text = (
                    ent["word"].replace("\u2581", " ").replace("Ġ", " ").strip()
                )
                if len(clean_text) < 1:
                    continue

                start_idx = ent.get("start")
                end_idx = ent.get("end")

                # Fallback to manual string matching if tokenizer strips offsets
                if start_idx is None or end_idx is None:
                    start_idx = text.find(clean_text)
                    end_idx = start_idx + len(clean_text) if start_idx != -1 else 0
                    start_idx = max(0, start_idx)

                # Strict cleaning: Remove trailing/leading punctuation
                stripped_text = clean_text.strip(".,:;() ")
                if not stripped_text:
                    continue

                if stripped_text != clean_text:
                    start_idx = text.find(stripped_text, start_idx)
                    end_idx = start_idx + len(stripped_text)

                entities.append(
                    {
                        "text": stripped_text,
                        "label": label,
                        "span": (start_idx, end_idx),
                    }
                )
    else:
        # 2. Clean Dictionary-Driven Rule-based Fallback
        FALLBACK_PATTERNS = {
            "DATE": r"\b(ngày\s+\d{1,2}(?:(?:/|-|tháng)\s*\d{1,2})?(?:(?:/|-|năm)\s*\d{4})?|\d+\s+(?:tháng|năm|ngày)|hàng\s+(?:tháng|năm|quý))\b",
            "MONEY": r"\b(\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?\s*(?:VNĐ|VND|đồng(?: Việt Nam)?|USD|usd|đ))\b",
            "RATE": r"(?<!phạt\s)(?<!phạt mức\s)\b(\d+(?:\.\d+)?\s*%)\b",
            "PENALTY": r"(?:\bphạt|\bđền bù|\bbồi thường)\s*(?:gấp đôi|gấp ba|\d+(?:\.\d+)?\s*%|\d{1,3}(?:[,.]\d{3})*\s*(?:VNĐ|VND|đồng|USD|đ))",
            "PARTY": r"\b(?:Bên|Người(?: lao động| sử dụng lao động| thuê| mua)?|Khách hàng|Công ty|Ngân hàng)\s*(?:[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ][\w]*\s*)*",
            "LAW": r"\b(?:Luật|Nghị định|Thông tư|Quyết định|Khoản|Điều)\s+[\w\s\./-]+\b",
        }

        for label, pattern in FALLBACK_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "text": match.group(0).strip(),
                        "label": label,
                        "span": match.span(),
                    }
                )

    return entities


def extract_entities(text: str) -> list[dict]:
    """Assignment NER requirement: Filter out OBJECT and PREDICATE."""
    raw = extract_ultra_entities(text)
    # Use ['label'] instead of ['entity_group'] to match the standardized output
    return [e for e in raw if e["label"] not in ["OBJECT", "PREDICATE"]]
