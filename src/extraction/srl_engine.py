import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "models/ultra_srl"


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
    Inference logic for the Heterogeneous Stacked BiRNN SRL model.
    Utilizes pre-extracted features from Assignment 1.
    Maintains exact interface: returns {"predicate": str, "roles": dict}
    """
    if not text or not text.strip():
        return {"predicate": "N/A", "roles": {}}

    model, tokenizer = get_srl_model()

    if model and model != "fallback":
        try:
            # 1. Feature Alignment: Map Stanza deps and NER labels to ViDeBERTa tokens
            # (Handling the 'thanh_toán' vs 'thanh', 'toán' mismatch using "First-token dominance")

            # 2. Convert to Tensors (input_ids, ner_ids, dep_ids, p_ner_ids)

            # 3. Model Forward Pass
            # with torch.no_grad():
            #     logits = model(input_ids, attention_mask, ner_ids, dep_ids, p_ner_ids)
            #     preds = torch.argmax(logits, dim=-1)

            # 4. Decode predictions into roles dictionary
            # Placeholder return demonstrating expected success path
            return {
                "predicate": "N/A",
                "roles": {
                    "Status": "DL Model Placeholder. Needs alignment logic implementation."
                },
            }
        except Exception as e:
            print(f"SRL Inference Error: {e}")
            # Fall through to fallback

    # --- FALLBACK LOGIC (If model isn't trained yet) ---
    roles = {}
    predicate = "N/A"

    # Simple Copula Fallback
    text_lower = text.lower()
    if " là " in text_lower and any(
        kw in text_lower for kw in ["bên", "người", "địa chỉ", "tổng"]
    ):
        predicate = "là"
        party_ents = [e["text"] for e in entities if e["label"] == "PARTY"]
        if len(party_ents) >= 1:
            roles["Theme"] = party_ents[0]
        if len(party_ents) >= 2:
            roles["Attribute"] = party_ents[1]

    # Simple Dependency Fallback
    elif dependencies:
        root_token = next(
            (d for d in dependencies if d.get("relation") == "root"), None
        )
        if root_token:
            predicate = root_token.get("token")

            # Find nsubj for Agent
            for d in dependencies:
                if d.get("relation") == "nsubj" and d.get(
                    "head_index"
                ) == root_token.get("id"):
                    # Quick mapping: find which entity this token belongs to
                    for ent in entities:
                        if d.get("token") in ent["text"]:
                            roles["Agent"] = ent["text"]
                            break

    # General Entity Mapping Fallback
    for e in entities:
        if e["label"] == "DATE" and "Time" not in roles:
            roles["Time"] = e["text"]
        elif e["label"] == "MONEY" and "Theme" not in roles:
            roles["Theme"] = e["text"]
        elif e["label"] in ["RATE", "PENALTY"]:
            roles["Penalty_Rate"] = e["text"]

    return {"predicate": predicate, "roles": roles}
