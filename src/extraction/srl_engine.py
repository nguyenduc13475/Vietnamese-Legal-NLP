import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "models/ultra_srl"
# Mappings for structural feature indexing (must match training)
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


class SRLStructuralSubmodel(nn.Module):
    def __init__(
        self,
        ner_vocab_size=10,
        dep_vocab_size=30,
        parent_ner_vocab_size=10,
        embed_dim=32,
    ):
        super().__init__()
        self.ner_emb = nn.Embedding(ner_vocab_size, embed_dim)
        self.dep_emb = nn.Embedding(dep_vocab_size, embed_dim)
        self.p_ner_emb = nn.Embedding(parent_ner_vocab_size, embed_dim)

    def forward(self, ner_ids, dep_ids, p_ner_ids):
        return torch.cat(
            [self.ner_emb(ner_ids), self.dep_emb(dep_ids), self.p_ner_emb(p_ner_ids)],
            dim=-1,
        )


class JointSRLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_model = AutoModel.from_pretrained("Fsoft-AIC/videberta-xsmall")
        self.structural_model = SRLStructuralSubmodel()
        # 384 (ViDeBERTa hidden size) + 96 (3 * 32 structural embeddings)
        self.bilstm = nn.LSTM(
            input_size=384 + 96, hidden_size=128, bidirectional=True, batch_first=True
        )
        self.classifier = nn.Linear(256, 11)  # 10 roles + O

    def forward(self, input_ids, attention_mask, ner_ids, dep_ids, p_ner_ids):
        # O1: Semantic features
        sem_out = self.semantic_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = sem_out.last_hidden_state

        # O2: Structural features
        struct_output = self.structural_model(ner_ids, dep_ids, p_ner_ids)

        # Concat O1 and O2
        combined_features = torch.cat([sequence_output, struct_output], dim=-1)

        # BiLSTM Contextualization
        lstm_out, _ = self.bilstm(combined_features)

        # Classification
        logits = self.classifier(lstm_out)
        return logits


_srl_model = None
_srl_tokenizer = None


def get_srl_model():
    """Lazy load the Joint SRL model."""
    global _srl_model, _srl_tokenizer
    if _srl_model is None:
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
            try:
                _srl_tokenizer = AutoTokenizer.from_pretrained(
                    "Fsoft-AIC/videberta-xsmall"
                )
                _srl_model = JointSRLModel()
                _srl_model.load_state_dict(
                    torch.load(
                        os.path.join(MODEL_PATH, "pytorch_model.bin"),
                        map_location="cpu",
                    )
                )
                _srl_model.eval()
            except Exception as e:
                print(f"Warning: Could not load SRL model. Error: {e}")
                _srl_model = "fallback"
        else:
            _srl_model = "fallback"
    return _srl_model, _srl_tokenizer


def extract_srl(
    text: str, entities: list, dependencies: list = None, np_chunks: list = None
) -> dict:
    """
    [Task 2.2] Heterogeneous stacked BiRNN SRL.
    Merges ViDeBERTa embeddings (O1) with Structural embeddings (O2).
    """
    model, tokenizer = get_srl_model()
    if model == "fallback":
        return {"predicate": "N/A", "roles": {}}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_offsets_mapping=True,
    )
    offsets = inputs["offset_mapping"][0]
    seq_len = inputs["input_ids"].shape[1]

    # Build O2 feature tensors
    ner_ids = torch.zeros(seq_len, dtype=torch.long)
    dep_ids = torch.zeros(seq_len, dtype=torch.long)
    p_ner_ids = torch.zeros(seq_len, dtype=torch.long)

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue

        # 1. Map NER Type
        for ent in entities:
            if start >= ent["span"][0] and end <= ent["span"][1]:
                ner_ids[i] = NER_MAP.get(ent["label"], 0)
                break

        # 2. Map Dependency and Parent NER (Token Alignment)
        if dependencies:
            for d in dependencies:
                # Use a sliding window match or character span check
                # to align Stanza's word tokens with DeBERTa's subword offsets
                d_token_clean = d["token"].replace("_", " ")
                d_start = text.find(d_token_clean, max(0, start - 5))
                d_end = d_start + len(d_token_clean)

                if start >= d_start and end <= d_end:
                    dep_ids[i] = DEP_MAP.get(d["relation"], 0)

                    # Contextual Enrichment: Find the NER type of the HEAD token
                    head_idx = d["head_index"]
                    parent = next(
                        (x for x in dependencies if x["id"] == head_idx), None
                    )
                    if parent:
                        p_token_clean = parent["token"].replace("_", " ")
                        # Find parent's NER label from our entities list
                        p_ent = next(
                            (e for e in entities if p_token_clean in e["text"]), None
                        )
                        if p_ent:
                            # Map complex labels like 'B-PARTY' to base 'PARTY' index
                            base_label = (
                                p_ent["label"].replace("B-", "").replace("I-", "")
                            )
                            p_ner_ids[i] = NER_MAP.get(base_label, 0)
                    break

    with torch.no_grad():
        logits = model(
            inputs["input_ids"],
            inputs["attention_mask"],
            ner_ids.unsqueeze(0),
            dep_ids.unsqueeze(0),
            p_ner_ids.unsqueeze(0),
        )
        preds = torch.argmax(logits, dim=-1)[0].tolist()

    # Aggregate tokens into roles
    roles = {}
    predicate = "N/A"
    for i, p_id in enumerate(preds):
        label = ID2SRL.get(p_id, "OTHER")
        if label == "OTHER":
            continue

        token_text = tokenizer.decode([inputs["input_ids"][0][i]]).replace(" ", "")
        if label == "PREDICATE":
            predicate = (
                token_text if predicate == "N/A" else predicate + " " + token_text
            )
        else:
            roles[label] = roles.get(label, "") + " " + token_text

    return {
        "predicate": predicate.strip(),
        "roles": {k: v.strip() for k, v in roles.items()},
    }
