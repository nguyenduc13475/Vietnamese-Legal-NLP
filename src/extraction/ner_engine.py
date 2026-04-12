import os
import re

from transformers import pipeline

# Load fine-tuned PhoBERT NER model if available
NER_MODEL_PATH = "./models/fine_tuned_ner"
ner_pipeline = None

if os.path.exists(NER_MODEL_PATH) and len(os.listdir(NER_MODEL_PATH)) > 0:
    import torch
    from transformers import AutoTokenizer

    try:
        device_id = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(
            NER_MODEL_PATH, clean_up_tokenization_spaces=True, model_max_length=256
        )
        ner_pipeline = pipeline(
            "token-classification",
            model=NER_MODEL_PATH,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            ignore_labels=["O"],
            device=device_id,
        )
    except Exception as e:
        print(f"Warning: Could not load NER model from {NER_MODEL_PATH}. Error: {e}")


def extract_entities(text: str) -> list[dict]:
    """
    Custom NER for Contracts.
    Prioritize using the fine-tuned PhoBERT model.
    If the model is not trained, fallback to Rule-based extraction.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    entities = []

    if ner_pipeline:
        # 1. Machine Learning Inference
        ml_entities = ner_pipeline(text)

        for ent in ml_entities:
            # simple aggregation strategy returns 'entity_group'
            label = ent.get("entity_group", "")
            if label:
                clean_text = ent["word"].replace("_", " ").replace("@@", "").strip()
                # Ensure we don't save empty/junk strings as entities
                if len(clean_text) < 1:
                    continue

                # PhoBERT lacks a Fast Tokenizer, so start/end offsets are often None.
                # We fallback to manual string matching to find the span.
                start_idx = ent.get("start")
                end_idx = ent.get("end")

                if start_idx is None or end_idx is None:
                    start_idx = text.find(clean_text)
                    end_idx = start_idx + len(clean_text) if start_idx != -1 else 0
                    start_idx = max(0, start_idx)

                entities.append(
                    {
                        "text": clean_text,
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
