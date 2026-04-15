import os
import pickle

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize

TFIDF_MODEL_PATH = "models/fine_tuned/intent_model.pkl"
VECTORIZER_PATH = "models/fine_tuned/intent_vectorizer.pkl"
TRANSFORMER_MODEL_PATH = "models/fine_tuned_intent_transformer"

_intent_model = None
_intent_tokenizer = None
_intent_config = None
_ml_model = None
_vectorizer = None


# --- Model Definitions (Local for load stability) ---
class RobustIntentModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.base_model.roberta(
            input_ids, attention_mask=attention_mask, return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(sequence_output))
        logits /= len(self.dropouts)
        return {"logits": logits}


def get_transformer_model():
    global _intent_model, _intent_tokenizer, _intent_config
    if _intent_model is None:
        if os.path.exists(TRANSFORMER_MODEL_PATH) and "pytorch_model.bin" in os.listdir(
            TRANSFORMER_MODEL_PATH
        ):
            try:
                _intent_tokenizer = AutoTokenizer.from_pretrained(
                    TRANSFORMER_MODEL_PATH
                )
                _intent_config = AutoConfig.from_pretrained(TRANSFORMER_MODEL_PATH)
                raw_model = AutoModelForSequenceClassification.from_config(
                    _intent_config
                )
                _intent_model = RobustIntentModel(raw_model)

                weights = torch.load(
                    os.path.join(TRANSFORMER_MODEL_PATH, "pytorch_model.bin"),
                    map_location="cpu",
                )
                _intent_model.base_model.load_state_dict(weights)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _intent_model.to(device)
                _intent_model.eval()
            except Exception as e:
                print(f"Warning: Intent Transformer load failed: {e}")
                _intent_model = "fallback"
        else:
            _intent_model = "fallback"
    return _intent_model, _intent_tokenizer, _intent_config


def get_ml_model():
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
    model, tokenizer, config = get_transformer_model()
    if model != "fallback" and model is not None:
        try:
            device = next(model.parameters()).device
            segmented_text = word_tokenize(text, format="text")
            inputs = tokenizer(
                segmented_text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                idx = torch.argmax(outputs["logits"], dim=1).item()
                return config.id2label[idx]
        except Exception:
            pass

    # 2. Try ML Model
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
