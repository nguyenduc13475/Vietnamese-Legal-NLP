import os
import pickle

import torch
from transformers import AutoTokenizer
from underthesea import word_tokenize

from src.utils.model_loader import load_robust_classification_model

TFIDF_MODEL_PATH = "models/intent_regression/intent_model.pkl"
VECTORIZER_PATH = "models/intent_regression/intent_vectorizer.pkl"
TRANSFORMER_MODEL_PATH = "models/intent_transformer"

_intent_model = None
_intent_tokenizer = None
_ml_model = None
_vectorizer = None

# Classes sorted alphabetically as done dynamically during training
INTENT_LABELS = [
    "Obligation",
    "Other",
    "Prohibition",
    "Right",
    "Termination Condition",
]
ID2INTENT = {i: label for i, label in enumerate(INTENT_LABELS)}


def get_transformer_model():
    """Lazy load the Intent Transformer model from local path."""
    global _intent_model, _intent_tokenizer
    if _intent_model is None:
        if os.path.exists(TRANSFORMER_MODEL_PATH):
            try:
                _intent_tokenizer = AutoTokenizer.from_pretrained(
                    TRANSFORMER_MODEL_PATH
                )
                _intent_model = load_robust_classification_model(
                    TRANSFORMER_MODEL_PATH,
                    num_labels=len(INTENT_LABELS),
                    is_token_level=False,
                )
            except Exception as e:
                print(f"Warning: Intent Transformer load failed: {e}")

        if _intent_model is None:
            _intent_model = "fallback"

    return _intent_model, _intent_tokenizer


def get_ml_model():
    """Lazy load the ML fallback model (TF-IDF + Logistic Regression)."""
    global _ml_model, _vectorizer
    if _ml_model is None:
        if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(TFIDF_MODEL_PATH, "rb") as f:
                _ml_model = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                _vectorizer = pickle.load(f)
    return _ml_model, _vectorizer


def classify_intent(text: str) -> str:
    """
    Classifies the legal intent of a clause.
    Logic: Transformer -> ML Model (TF-IDF) -> Keywords.
    """
    if not text or not text.strip():
        return "Other"

    # 1. Try Deep Learning (PhoBERT)
    model, tokenizer = get_transformer_model()
    if model != "fallback" and model is not None:
        try:
            device = next(model.parameters()).device
            segmented_text = word_tokenize(text, format="text")

            inputs = tokenizer(
                segmented_text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs)["logits"]
                idx = torch.argmax(logits, dim=1).item()
                return ID2INTENT[idx]
        except Exception as e:
            print(f"Transformer inference error: {e}")

    # 2. Try ML Model (TF-IDF Baseline)
    ml_model, vectorizer = get_ml_model()
    if ml_model and vectorizer:
        try:
            segmented_text = word_tokenize(text, format="text")
            X = vectorizer.transform([segmented_text])
            return ml_model.predict(X)[0]
        except Exception:
            pass

    # 3. Rule-based fallback
    INTENT_KEYWORDS = {
        "Prohibition": ["không được", "cấm", "nghiêm cấm", "không có quyền"],
        "Right": ["có quyền", "được phép", "toàn quyền", "có thể hưởng"],
        "Termination Condition": [
            "chấm dứt",
            "hủy bỏ",
            "hết hiệu lực",
            "vi phạm hợp đồng",
        ],
        "Obligation": ["phải", "có trách nhiệm", "cam kết", "nghĩa vụ", "thanh toán"],
    }
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent

    return "Other"
