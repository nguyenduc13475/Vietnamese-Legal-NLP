import os

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from underthesea import word_tokenize

from src.models.robust_base import JointSRLModel, RobustSRLModel
from src.preprocessing.parser import get_pipeline
from src.utils.constants import (
    DEP_RELATION_MAP,
    NER_CATEGORY_MAP,
    NER_LABEL_MAP,
    SRL_ROLE_MAP,
)

MODEL_PATH = "models/srl"
BASE_MODEL_NAME = "vinai/phobert-base"

_srl_model = None
_srl_tokenizer = None

ID2SRL = {v: k for k, v in SRL_ROLE_MAP.items()}
ID2NER = {v: k for k, v in NER_LABEL_MAP.items()}


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


def extract_srl(text: str, entities: list = None) -> dict:
    model, tokenizer = get_srl_model()
    if model == "fallback" or model is None:
        return {"predicate": "N/A", "roles": {}}

    device = next(model.parameters()).device
    text_segmented = word_tokenize(text, format="text")

    # 1. Stanza Analysis (Structural Source)
    nlp = get_pipeline()
    stanza_doc = nlp(text)
    stanza_words = stanza_doc.sentences[0].words if stanza_doc.sentences else []

    # 2. PhoBERT Tokenization (Semantic Target)
    encoding = tokenizer(
        text_segmented, return_tensors="pt", truncation=True, max_length=256
    ).to(device)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    # 3. Precise Character Alignment (Matching auto_annotate.py logic)
    text_stripped = "".join(text.split()).lower()

    def get_stripped_spans(word_list, is_stanza=False):
        spans = []
        curr = 0
        for w in word_list:
            # Stanza uses .text, Tokenizer uses token string
            raw = w.text if is_stanza else w
            clean = raw.replace("@@", "").replace("_", "").replace(" ", "").lower()
            if not clean:
                spans.append((-1, -1))
                continue
            start = text_stripped.find(clean, curr)
            if start != -1:
                end = start + len(clean)
                spans.append((start, end))
                curr = end
            else:
                spans.append((-1, -1))
        return spans

    token_spans = get_stripped_spans(tokens, is_stanza=False)
    stanza_spans = get_stripped_spans(stanza_words, is_stanza=True)

    # 4. Feature Extraction
    seq_len = len(tokens)
    ner_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    dep_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    p_ner_ids = torch.zeros(seq_len, dtype=torch.long, device=device)

    # Pre-map Stanza words to NER categories
    stanza_ner_cats = [0] * len(stanza_words)
    if entities:
        for i, sw in enumerate(stanza_words):
            sw_clean = sw.text.replace("_", " ").lower()
            for ent in entities:
                if sw_clean in ent["text"].lower():
                    stanza_ner_cats[i] = NER_CATEGORY_MAP.get(ent["label"], 0)
                    break

    for i, (tok_s, tok_e) in enumerate(token_spans):
        if tok_s == -1:
            continue

        # Match current token to an Entity
        for ent in entities:
            e_strip = "".join(ent["text"].split()).lower()
            e_start = text_stripped.find(e_strip)
            if e_start != -1 and tok_s >= e_start and tok_e <= e_start + len(e_strip):
                ner_ids[i] = NER_CATEGORY_MAP.get(ent["label"], 0)
                break

        # Match current token to Stanza word for Dependency/Parent
        best_w = -1
        max_ov = 0
        for w_idx, (w_s, w_e) in enumerate(stanza_spans):
            if w_s == -1:
                continue
            overlap = min(tok_e, w_e) - max(tok_s, w_s)
            if overlap > max_ov:
                max_ov, best_w = overlap, w_idx

        if best_w != -1:
            sw = stanza_words[best_w]
            dep_ids[i] = DEP_RELATION_MAP.get(sw.deprel, 0)
            p_idx = sw.head - 1
            if 0 <= p_idx < len(stanza_ner_cats):
                p_ner_ids[i] = stanza_ner_cats[p_idx]

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
    # Create a mask for the entire original text length
    # We store the label name at each character index
    char_label_mask = [None] * len(text)

    # Force Predicate from NER entities into mask
    if entities:
        for ent in entities:
            if ent["label"] == "PREDICATE":
                for m_idx in range(ent["span"][0], ent["span"][1]):
                    char_label_mask[m_idx] = "PREDICATE"

    for i, p_id in enumerate(preds):
        if tokens[i] in [tokenizer.bos_token, tokenizer.eos_token]:
            continue

        label = ID2SRL.get(p_id, "OTHER")
        if label == "OTHER":
            continue

        # Use token_spans from Step 3 which maps to text_stripped
        # We need to map it back to the REAL text (with spaces)
        ts, te = token_spans[i]
        if ts == -1:
            continue

        # Map stripped indices back to original text indices
        # This is the most robust way
        chars_found = 0
        real_start = -1
        real_end = -1

        for idx, char in enumerate(text):
            if not char.isspace():
                if chars_found == ts:
                    real_start = idx
                chars_found += 1
                if chars_found == te:
                    real_end = idx + 1
                    break

        if real_start != -1 and real_end != -1:
            for m_idx in range(real_start, real_end):
                char_label_mask[m_idx] = label

    # Now extract strings based on the mask
    extracted_roles = {}
    predicate = ""

    # Temporary storage to build strings
    current_label = None
    start_pos = -1

    for idx, label in enumerate(
        char_label_mask + [None]
    ):  # Extra None to flush last label
        if label != current_label:
            if current_label is not None:
                # Extract the exact substring from original text
                substring = text[start_pos:idx].strip()
                if substring:
                    if current_label == "PREDICATE":
                        predicate = (
                            substring
                            if predicate == "N/A"
                            else predicate + " " + substring
                        )
                    else:
                        if current_label not in extracted_roles:
                            extracted_roles[current_label] = substring
                        else:
                            extracted_roles[current_label] += " " + substring

            current_label = label
            start_pos = idx

    # Final cleanup of internal double spaces
    predicate = " ".join(predicate.split()) if predicate else "N/A"
    final_roles = {k: " ".join(v.split()) for k, v in extracted_roles.items()}

    return {"predicate": predicate, "roles": final_roles}
