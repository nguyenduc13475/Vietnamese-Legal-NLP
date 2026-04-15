import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "models/ultra_srl"
BASE_MODEL_NAME = "vinai/phobert-base"

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
SRL2ID = {v: k for k, v in ID2SRL.items()}


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
    def __init__(self, base_model=None):
        super().__init__()
        # Use PhoBERT as the semantic backbone
        self.semantic_model = (
            base_model if base_model else AutoModel.from_pretrained(BASE_MODEL_NAME)
        )
        self.config = self.semantic_model.config
        self.structural_model = SRLStructuralSubmodel()

        # 768 (PhoBERT) + 96 (Structural: 32*3)
        self.bilstm = nn.LSTM(
            input_size=768 + 96, hidden_size=256, bidirectional=True, batch_first=True
        )
        self.classifier = nn.Linear(512, 11)  # 10 roles + OTHER

    def forward(
        self,
        input_ids,
        attention_mask,
        ner_ids=None,
        dep_ids=None,
        p_ner_ids=None,
        **kwargs,
    ):
        sem_out = self.semantic_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = sem_out.last_hidden_state

        # Safety: If structural features are missing (common during generic training steps), use zeros
        if ner_ids is None:
            ner_ids = torch.zeros(
                (input_ids.shape[0], input_ids.shape[1]),
                dtype=torch.long,
                device=input_ids.device,
            )
        if dep_ids is None:
            dep_ids = torch.zeros_like(ner_ids)
        if p_ner_ids is None:
            p_ner_ids = torch.zeros_like(ner_ids)

        struct_output = self.structural_model(ner_ids, dep_ids, p_ner_ids)
        combined_features = torch.cat([sequence_output, struct_output], dim=-1)
        lstm_out, _ = self.bilstm(combined_features)
        logits = self.classifier(lstm_out)
        return {"logits": logits}


class RobustSRLModel(nn.Module):
    """Multi-sample dropout for SRL stability."""

    def __init__(self, joint_model):
        super().__init__()
        self.base_model = joint_model
        self.config = joint_model.config
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        ner_ids=None,
        dep_ids=None,
        p_ner_ids=None,
        labels=None,
        **kwargs,
    ):
        # We wrap the LSTM output or the input to the final classifier
        # For simplicity and effectiveness, we apply multi-dropout to the LSTM output
        sem_out = self.base_model.semantic_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = sem_out.last_hidden_state

        # Handle optional O2 features
        if ner_ids is None:
            ner_ids = torch.zeros_like(input_ids)
        if dep_ids is None:
            dep_ids = torch.zeros_like(input_ids)
        if p_ner_ids is None:
            p_ner_ids = torch.zeros_like(input_ids)

        struct_output = self.base_model.structural_model(ner_ids, dep_ids, p_ner_ids)
        combined = torch.cat([sequence_output, struct_output], dim=-1)
        lstm_out, _ = self.base_model.bilstm(combined)

        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(lstm_out))
        logits /= len(self.dropouts)
        return {"logits": logits}


_srl_model = None
_srl_tokenizer = None


def get_srl_model():
    """Lazy load the Joint SRL model."""
    global _srl_model, _srl_tokenizer
    if _srl_model is None:
        if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
            try:
                _srl_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
                base_joint = JointSRLModel()
                _srl_model = RobustSRLModel(base_joint)
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
