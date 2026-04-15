import os

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from underthesea import word_tokenize

from src.extraction.ner_engine import extract_ultra_entities
from src.models.robust_base import JointSRLModel, RobustSRLModel
from src.preprocessing.parser import get_pipeline

MODEL_PATH = "models/srl"
BASE_MODEL_NAME = "vinai/phobert-base"

_srl_model = None
_srl_tokenizer = None

# Mappings must perfectly match auto_annotate.py
NER_MAP = {
    "O": 0,
    "PARTY": 1,
    "MONEY": 2,
    "DATE": 3,
    "RATE": 4,
    "PENALTY": 5,
    "LAW": 6,
    "OBJECT": 7,
    "PREDICATE": 8,
}
DEP_MAP = {
    "root": 1,
    "nsubj": 2,
    "obj": 3,
    "iobj": 4,
    "obl": 5,
    "advcl": 6,
    "amod": 7,
    "nmod": 8,
    "compound": 9,
    "mark": 10,
    "advmod": 11,
    "xcomp": 12,
    "cc": 13,
    "conj": 14,
    "det": 15,
    "case": 16,
    "fixed": 17,
    "flat": 18,
    "punct": 19,
}
ID2SRL = {
    0: "OTHER",
    1: "AGENT",
    2: "RECIPIENT",
    3: "THEME",
    4: "NAME",
    5: "TIME",
    6: "CONDITION",
    7: "TRAIT",
    8: "LOCATION",
    9: "METHOD",
    10: "ABOUT",
}


def get_srl_model():
    global _srl_model, _srl_tokenizer
    if _srl_model is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(
            os.path.join(MODEL_PATH, "pytorch_model.bin")
        ):
            try:
                _srl_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
                sem_base = AutoModel.from_config(config)
                joint = JointSRLModel(sem_base)
                _srl_model = RobustSRLModel(joint)
                weights = torch.load(
                    os.path.join(MODEL_PATH, "pytorch_model.bin"), map_location="cpu"
                )
                _srl_model.base_model.load_state_dict(weights, strict=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _srl_model.to(device).eval()
            except Exception as e:
                print(f"Error loading SRL: {e}")
                _srl_model = "fallback"
        else:
            _srl_model = "fallback"
    return _srl_model, _srl_tokenizer


def extract_srl(
    text: str, entities: list = None, dependencies: list = None, np_chunks: list = None
) -> dict:
    model, tokenizer = get_srl_model()
    if model == "fallback" or model is None:
        return {"predicate": "N/A", "roles": {}}

    device = next(model.parameters()).device
    # 1. Get real-time features if not provided (Important for direct API calls)
    if entities is None:
        entities = extract_ultra_entities(text)

    # 2. Stanza Dependency Analysis
    nlp = get_pipeline()
    stanza_doc = nlp(text)
    stanza_words = stanza_doc.sentences[0].words if stanza_doc.sentences else []

    # 3. PhoBERT Tokenization and Manual Alignment
    # Standardize text for position tracking
    text_segmented = word_tokenize(text, format="text")
    encoding = tokenizer(
        text_segmented, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    seq_len = encoding["input_ids"].shape[1]
    ner_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    dep_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    p_ner_ids = torch.zeros(seq_len, dtype=torch.long, device=device)

    # Replicate auto_annotate.py character-level overlap logic
    text_stripped = "".join(text.split()).lower()

    # Track character spans for PhoBERT tokens
    token_spans = []
    cursor = 0
    for tok in tokens:
        clean_tok = tok.replace("@@", "").replace("_", "").lower()
        if (
            tok in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]
            or not clean_tok
        ):
            token_spans.append((-1, -1))
            continue
        start = text_stripped.find(clean_tok, cursor)
        if start != -1:
            end = start + len(clean_tok)
            token_spans.append((start, end))
            cursor = end
        else:
            token_spans.append((-1, -1))

    # Track character spans for Stanza words
    stanza_spans = []
    cursor = 0
    for sw in stanza_words:
        clean_w = sw.text.replace("_", "").lower()
        start = text_stripped.find(clean_w, cursor)
        if start != -1:
            end = start + len(clean_w)
            stanza_spans.append((start, end))
            cursor = end
        else:
            stanza_spans.append((-1, -1))

    # Map Entities to Stanza words for parent context
    stanza_ner_labels = ["O"] * len(stanza_words)
    for i, sw in enumerate(stanza_words):
        for ent in entities:
            # Simple substring match for entity mapping
            if sw.text.replace("_", " ") in ent["text"]:
                stanza_ner_labels[i] = ent["label"]
                break

    # 4. Feature Filling
    for i, (tok_s, tok_e) in enumerate(token_spans):
        if tok_s == -1:
            continue

        # NER ID for current token
        for ent in entities:
            ent_stripped = "".join(ent["text"].split()).lower()
            e_start = text_stripped.find(ent_stripped)
            if (
                e_start != -1
                and tok_s >= e_start
                and tok_e <= e_start + len(ent_stripped)
            ):
                ner_ids[i] = NER_MAP.get(ent["label"], 0)
                break

        # Dependency and Parent NER via Overlap
        best_w_idx = -1
        max_ov = 0
        for w_idx, (w_s, w_e) in enumerate(stanza_spans):
            if w_s == -1:
                continue
            ov = min(tok_e, w_e) - max(tok_s, w_s)
            if ov > max_ov:
                max_ov = ov
                best_w_idx = w_idx

        if best_w_idx != -1:
            sw = stanza_words[best_w_idx]
            dep_ids[i] = DEP_MAP.get(sw.deprel, 0)
            p_idx = sw.head - 1
            if 0 <= p_idx < len(stanza_ner_labels):
                p_ner_ids[i] = NER_MAP.get(stanza_ner_labels[p_idx], 0)

    # 5. Model Inference
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            ner_ids=ner_ids.unsqueeze(0),
            dep_ids=dep_ids.unsqueeze(0),
            p_ner_ids=p_ner_ids.unsqueeze(0),
        )
        preds = torch.argmax(outputs["logits"], dim=-1)[0].cpu().tolist()

    # 6. Result Aggregation
    roles = {}
    predicate = "N/A"
    for i, p_id in enumerate(preds):
        if tokens[i] in [tokenizer.bos_token, tokenizer.eos_token]:
            continue
        label = ID2SRL.get(p_id, "OTHER")
        if label == "OTHER":
            continue

        word = tokens[i].replace("@@", "").replace("_", " ").strip()
        if label == "PREDICATE":
            predicate = word if predicate == "N/A" else predicate + word
        else:
            roles[label] = (
                roles.get(label, "")
                + ("" if tokens[i].startswith("@@") else " ")
                + word
            ).strip()

    return {"predicate": predicate.strip(), "roles": roles}
