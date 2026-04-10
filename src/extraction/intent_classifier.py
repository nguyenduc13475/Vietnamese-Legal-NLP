import os
import pickle

import torch
from transformers import pipeline

TFIDF_MODEL_PATH = "models/fine_tuned/intent_model.pkl"
VECTORIZER_PATH = "models/fine_tuned/intent_vectorizer.pkl"
TRANSFORMER_MODEL_PATH = "models/fine_tuned_intent_transformer"

ml_model = None
vectorizer = None
transformer_pipeline = None

# Attempt to load Transformer first; if unavailable, fallback to TF-IDF.
if (
    os.path.exists(TRANSFORMER_MODEL_PATH)
    and len(os.listdir(TRANSFORMER_MODEL_PATH)) > 0
):
    try:
        device_id = 0 if torch.cuda.is_available() else -1
        transformer_pipeline = pipeline(
            "text-classification",
            model=TRANSFORMER_MODEL_PATH,
            tokenizer=TRANSFORMER_MODEL_PATH,
            truncation=True,
            max_length=256,
            device=device_id,
        )
    except Exception as e:
        print(f"Warning: Could not load Transformer Intent model. Error: {e}")

if (
    transformer_pipeline is None
    and os.path.exists(TFIDF_MODEL_PATH)
    and os.path.exists(VECTORIZER_PATH)
):
    with open(TFIDF_MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)


def classify_intent(text: str) -> str:
    """
    Classifies the legal intent of a clause.
    Prior Transformer -> ML Model (TF-IDF + LR) -> Rule-based.
    """
    if transformer_pipeline:
        result = transformer_pipeline(text)
        return result[0]["label"]

    if ml_model and vectorizer:
        X = vectorizer.transform([text])
        return ml_model.predict(X)[0]

    # Dictionary-based Fallback Heuristics
    INTENT_KEYWORDS = {
        "Prohibition": [
            "không được",
            "cấm",
            "nghiêm cấm",
            "không có quyền",
            "tuyệt đối không",
            "từ chối",
            "hạn chế",
        ],
        "Right": [
            "có quyền",
            "được phép",
            "toàn quyền",
            "có thể",
            "quyền lợi",
            "được hưởng",
        ],
        "Termination Condition": [
            "chấm dứt",
            "hủy bỏ",
            "đơn phương hủy",
            "hết hiệu lực",
            "vô hiệu",
            "đáo hạn",
            "thanh lý",
        ],
        "Obligation": [
            "phải",
            "có trách nhiệm",
            "cam kết",
            "nghĩa vụ",
            "đồng ý",
            "chịu trách nhiệm",
            "bắt buộc",
            "bị phạt",
            "bồi thường",
            "đảm bảo",
            "thanh toán",
        ],
    }

    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent

    return "Other"
