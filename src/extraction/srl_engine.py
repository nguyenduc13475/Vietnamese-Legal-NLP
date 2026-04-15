import os

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from underthesea import word_tokenize

from src.models.robust_base import JointSRLModel, MultiSampleDropoutWrapper

MODEL_PATH = "models/srl"
BASE_MODEL_NAME = "vinai/phobert-base"

_srl_model = None
_srl_tokenizer = None

# Mappings for structural feature indexing (must match training exactly)
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
    "B-PARTY": 1,
    "I-PARTY": 1,
    "B-MONEY": 2,
    "I-MONEY": 2,
    "B-DATE": 3,
    "I-DATE": 3,
    "B-RATE": 4,
    "I-RATE": 4,
    "B-PENALTY": 5,
    "I-PENALTY": 5,
    "B-LAW": 6,
    "I-LAW": 6,
    "B-OBJECT": 7,
    "I-OBJECT": 7,
    "B-PREDICATE": 8,
    "I-PREDICATE": 8,
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
SRL2ID = {v: k for k, v in ID2SRL.items()}


def get_srl_model():
    """Lazy load the Joint SRL model prioritizing local artifacts."""
    global _srl_model, _srl_tokenizer
    if _srl_model is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(
            os.path.join(MODEL_PATH, "pytorch_model.bin")
        ):
            try:
                _srl_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                config_src = (
                    MODEL_PATH
                    if os.path.exists(os.path.join(MODEL_PATH, "config.json"))
                    else BASE_MODEL_NAME
                )
                backbone_config = AutoConfig.from_pretrained(config_src)
                sem_base = AutoModel.from_config(backbone_config)

                joint = JointSRLModel(sem_base)
                _srl_model = MultiSampleDropoutWrapper(joint)

                weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
                _srl_model.base_model.load_state_dict(
                    torch.load(weights_path, map_location="cpu"), strict=False
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _srl_model.to(device).eval()
            except Exception as e:
                print(f"Warning: SRL model load failed: {e}")
                _srl_model = "fallback"
        else:
            _srl_model = "fallback"
    return _srl_model, _srl_tokenizer


def extract_srl(
    text: str, entities: list, dependencies: list = None, np_chunks: list = None
) -> dict:
    """
    [Task 2.2] Semantic Role Labeling using aligned Pre-tokenization.
    """
    model, tokenizer = get_srl_model()
    if model == "fallback" or model is None:
        return {"predicate": "N/A", "roles": {}}

    device = next(model.parameters()).device

    # Synchronized Tokenization
    raw_tokens = word_tokenize(text)

    # 1. Dependency Parsing via Stanza (Pre-tokenized mode)
    from src.preprocessing.parser import get_pipeline

    nlp = get_pipeline()
    doc = nlp([raw_tokens])
    stanza_words = doc.sentences[0].words

    # 2. PhoBERT Encoding with word-to-subword alignment
    encoding = tokenizer(
        raw_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    word_ids = encoding.word_ids(batch_index=0)

    seq_len = encoding["input_ids"].shape[1]
    ner_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    dep_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    p_ner_ids = torch.zeros(seq_len, dtype=torch.long, device=device)

    # 3. Features alignment using word_ids mapping
    for i, w_idx in enumerate(word_ids):
        if w_idx is None:
            continue  # Skip BOS/EOS/PAD

        # Current token text
        token_text = raw_tokens[w_idx]

        # NER Feature (from trained NER engine output)
        for ent in entities:
            if token_text in ent["text"]:
                ner_ids[i] = NER_MAP.get(ent["label"], 0)
                break

        # Dependency Feature
        sw = stanza_words[w_idx]
        dep_ids[i] = DEP_MAP.get(sw.deprel, 0)

        # Parent NER Feature (Head's Entity type)
        p_idx = sw.head - 1  # Stanza head is 1-based
        if p_idx >= 0:
            p_text = raw_tokens[p_idx]
            for ent in entities:
                if p_text in ent["text"]:
                    p_ner_ids[i] = NER_MAP.get(ent["label"], 0)
                    break

    # 4. Model Inference
    with torch.no_grad():
        outputs = model(
            encoding["input_ids"],
            encoding["attention_mask"],
            ner_ids=ner_ids.unsqueeze(0),
            dep_ids=dep_ids.unsqueeze(0),
            p_ner_ids=p_ner_ids.unsqueeze(0),
        )
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    # 5. Role Aggregation (merging subwords and adjacent roles)
    roles = {}
    predicate = "N/A"

    for i, p_id in enumerate(preds):
        w_idx = word_ids[i]
        if w_idx is None:
            continue  # Ignore special tokens

        label = ID2SRL.get(p_id, "OTHER")
        if label == "OTHER":
            continue

        # Decode the specific subword
        subword_text = tokenizer.decode([encoding["input_ids"][0][i]]).replace(" ", "")

        if label == "PREDICATE":
            if predicate == "N/A":
                predicate = subword_text
            else:
                predicate += " " + subword_text
        else:
            # Clean up the role text (merging subwords with "_" logic for Vietnamese)
            current_val = roles.get(label, "")
            if current_val == "":
                roles[label] = subword_text
            else:
                # If it belongs to the same word (subword), don't add space
                # else add space for new word
                roles[label] += (
                    "" if subword_text.startswith("@@") else " "
                ) + subword_text.replace("@@", "")

    # Final cleanup of role strings
    cleaned_roles = {k: v.replace("@@", "").strip() for k, v in roles.items()}

    return {"predicate": predicate.replace("@@", "").strip(), "roles": cleaned_roles}
