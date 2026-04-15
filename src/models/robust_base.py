import torch
import torch.nn as nn


class MultiSampleDropoutWrapper(nn.Module):
    """Generic wrapper for Sequence and Token classification with Multi-Sample Dropout."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model.roberta(
            input_ids, attention_mask=attention_mask, return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(sequence_output))
        logits /= len(self.dropouts)
        return {"logits": logits}


class SRLStructuralSubmodel(nn.Module):
    def __init__(
        self,
        ner_vocab_size=20,
        dep_vocab_size=50,
        parent_ner_vocab_size=20,
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
    def __init__(self, semantic_base):
        super().__init__()
        self.semantic_model = semantic_base
        self.config = semantic_base.config
        self.structural_model = SRLStructuralSubmodel()
        # 768 (PhoBERT) + 96 (Structural: 32*3)
        self.bilstm = nn.LSTM(
            input_size=768 + 96, hidden_size=256, bidirectional=True, batch_first=True
        )
        self.classifier = nn.Linear(512, 11)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        ner_ids=None,
        dep_ids=None,
        p_ner_ids=None,
        **kwargs,
    ):
        sem_out = self.semantic_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = sem_out.last_hidden_state

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
        combined = torch.cat([sequence_output, struct_output], dim=-1)
        lstm_out, _ = self.bilstm(combined)
        return {"logits": self.classifier(lstm_out)}


class RobustSRLModel(nn.Module):
    """Wrapper specifically for the Joint SRL architecture to handle dropout during training."""

    def __init__(self, base_joint_model):
        super().__init__()
        self.base_model = base_joint_model
        self.config = base_joint_model.config
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i + 1)) for i in range(5)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        ner_ids=None,
        dep_ids=None,
        p_ner_ids=None,
        **kwargs,
    ):
        # We perform multi-sample dropout on the output of the BiLSTM inside the base_model
        sem_out = self.base_model.semantic_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = sem_out.last_hidden_state

        struct_output = self.base_model.structural_model(ner_ids, dep_ids, p_ner_ids)
        combined = torch.cat([sequence_output, struct_output], dim=-1)
        lstm_out, _ = self.base_model.bilstm(combined)

        logits = 0
        for dropout in self.dropouts:
            logits += self.base_model.classifier(dropout(lstm_out))
        logits /= len(self.dropouts)
        return {"logits": logits}
