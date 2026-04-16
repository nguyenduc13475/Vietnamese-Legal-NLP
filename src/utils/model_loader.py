import os

import safetensors.torch
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from src.models.robust_base import (
    JointSRLModel,
    MultiSampleDropoutWrapper,
    RobustSRLModel,
)


def load_robust_classification_model(model_path, num_labels, is_token_level=True):
    """Manually reconstructs the model prioritizing local config to avoid Hub/Offline crashes."""
    # Priority 1: Load config from the fine-tuned directory
    # Priority 2: Fallback to base model name (will fail if offline and not cached)
    config_path = (
        model_path
        if os.path.exists(os.path.join(model_path, "config.json"))
        else "vinai/phobert-base"
    )

    try:
        # Priority: Load strictly from local path without checking Hub
        config = AutoConfig.from_pretrained(
            config_path, num_labels=num_labels, local_files_only=True
        )
    except Exception:
        # If the specific path fails, try the base cache without poking internet
        config = AutoConfig.from_pretrained(
            "vinai/phobert-base", num_labels=num_labels, local_files_only=True
        )

    if "srl" in model_path.lower():
        # SRL uses a custom Joint architecture with BiLSTM
        semantic_base = AutoModel.from_config(config)
        base = JointSRLModel(semantic_base)
        model = RobustSRLModel(base)
    elif is_token_level:
        base = AutoModelForTokenClassification.from_config(config)
        model = MultiSampleDropoutWrapper(base)
    else:
        base = AutoModelForSequenceClassification.from_config(config)
        model = MultiSampleDropoutWrapper(base)

    # 1. Locate the weight file
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safe_path = os.path.join(model_path, "model.safetensors")

    state_dict = None
    if os.path.exists(safe_path):
        state_dict = safetensors.torch.load_file(safe_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        print(f"Error: No weight files found in {model_path}")
        return None

    # 2. Map keys to the Robust wrapper (base_model. prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("base_model."):
            new_state_dict[f"base_model.{k}"] = v
        else:
            new_state_dict[k] = v

    # 3. Load into model (Strict=False for SRL because of structural embeddings)
    model.load_state_dict(new_state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()
