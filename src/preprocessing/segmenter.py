import os
import re

import torch
from transformers import AutoTokenizer
from underthesea import word_tokenize

from src.utils.model_loader import load_robust_classification_model

MODEL_PATH = "models/segmenter"
_segment_model = None
_segment_tokenizer = None


def get_segment_resources():
    """Lazy load the DL Segmenter model and tokenizer from local path."""
    global _segment_model, _segment_tokenizer
    if _segment_model is None:
        if os.path.exists(MODEL_PATH):
            try:
                _segment_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                _segment_model = load_robust_classification_model(
                    MODEL_PATH, num_labels=6, is_token_level=True
                )
            except Exception as e:
                print(f"Error loading segmenter resources: {e}")

        if _segment_model is None:
            print(
                "Segmenter model not found or failed to load. Falling back to line-based."
            )
            _segment_model = "fallback"

    return _segment_model, _segment_tokenizer


def segment_clauses(text: str) -> list[dict]:
    """
    Splits text into clauses. Uses ViDeBERTa to identify components A,B,C,D,E
    and reconstructs them into valid legal sentences.
    """
    raw_lines = text.split("\n")
    clauses = []
    current_context = "General"
    current_aliases = "[]"

    model, tokenizer = get_segment_resources()

    bullet_pattern = re.compile(
        r"^([a-zđ][\)\.]|\d+(?:\.\d+)*[\.\):]?|[IVXLCDM]+[\.\)]|[-+])\s+", re.IGNORECASE
    )

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[ALIASES]"):
            current_aliases = line.replace("[ALIASES]", "", 1).strip()
            continue

        is_title = False
        if line.startswith("[TITLE]"):
            is_title = True
            line = line.replace("[TITLE]", "", 1).strip()

        ctx_match = re.match(r"^\[(.*?)\]\s*(.*)", line)
        if ctx_match:
            current_context = ctx_match.group(1).strip()
            line = ctx_match.group(2).strip()

        clean_line = bullet_pattern.sub("", line).strip()
        if len(clean_line) <= 5:
            continue

        if model != "fallback":
            try:
                device = next(model.parameters()).device

                # 1. Temporarily strip trailing dots to match training data distribution
                line_to_process = clean_line.rstrip(". ")

                # 2. Pre-segment text for PhoBERT consistency
                segmented_input = word_tokenize(line_to_process, format="text")

                inputs = tokenizer(
                    segmented_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(device)

                with torch.no_grad():
                    logits = model(**inputs)["logits"]
                    preds = torch.argmax(logits, dim=2)[0].cpu().numpy()

                # Kiểm tra tính hợp lệ của chuỗi nhãn
                valid_labels = [p for p in preds if 1 <= p <= 5]
                # Nếu nhãn bị lộn xộn (ví dụ [2, 3, 2]) hoặc không có nhãn nào được dự đoán
                if not valid_labels or not all(
                    valid_labels[i] <= valid_labels[i + 1]
                    for i in range(len(valid_labels) - 1)
                ):
                    # Force nhảy xuống phần fallback bằng cách raise lỗi hoặc skip
                    raise ValueError("Inconsistent label sequence detected")

                input_ids = inputs["input_ids"][0].cpu().tolist()

                # 3. Prepare alignment mapping to force strict substring recovery
                # We map indices from a 'no-space' version back to the original line
                collapsed_orig = ""
                idx_map = []
                for idx, char in enumerate(line_to_process):
                    if not char.isspace():
                        collapsed_orig += char
                        idx_map.append(idx)

                def get_original_substring(reconstructed_text):
                    # Remove all spaces/underscores from model output to find anchor
                    target = reconstructed_text.replace(" ", "").replace("_", "")
                    if not target:
                        return ""

                    start_in_collapsed = collapsed_orig.find(target)
                    if start_in_collapsed == -1:
                        # Fallback: if not found, return cleaned reconstruction
                        return reconstructed_text.replace("_", " ").strip()

                    end_in_collapsed = start_in_collapsed + len(target) - 1
                    actual_start = idx_map[start_in_collapsed]
                    actual_end = idx_map[end_in_collapsed] + 1
                    return line_to_process[actual_start:actual_end].strip()

                # 4. Collect and Align blocks
                blocks_ids = {i: [] for i in range(1, 6)}
                for i, p_id in enumerate(preds):
                    if 1 <= p_id <= 5:
                        tid = input_ids[i]
                        if tid not in tokenizer.all_special_ids:
                            blocks_ids[p_id].append(tid)

                # Use native decode, then project back to original substring
                b = {}
                for k, v in blocks_ids.items():
                    decoded = tokenizer.decode(v)
                    b[k] = get_original_substring(decoded)

                # Type detection logic: construction occur only if at least C or D non empty
                if not b[3] and not b[4]:
                    c_list = [clean_line]
                elif b[5] or (preds == 5).any():
                    c_list = [f"{b[1]} {b[2]} {b[3]}", f"{b[1]} {b[4]} {b[3]} {b[5]}"]
                else:
                    c_list = [f"{b[1]} {b[2]} {b[4]}", f"{b[1]} {b[3]} {b[4]}"]

                for idx, c_text in enumerate(c_list):
                    # Clean double spaces and ensure trailing dot
                    final_text = " ".join(c_text.split()).strip()
                    if final_text:
                        final_text = final_text[0].upper() + final_text[1:]
                    if len(final_text) > 2:
                        # Ensure we don't double dot if original substring already had one
                        out_text = (
                            final_text if final_text.endswith(".") else final_text + "."
                        )
                        clauses.append(
                            {
                                "text": out_text,
                                "context": current_context,
                                "is_title": is_title if idx == 0 else False,
                                "aliases": current_aliases,
                            }
                        )
                continue
            except Exception as e:
                print(f"DL Segmentation inference error: {e}")

        # Fallback if model fails
        clauses.append(
            {
                "text": clean_line if clean_line.endswith(".") else clean_line + ".",
                "context": current_context,
                "is_title": is_title,
                "aliases": current_aliases,
            }
        )

    return clauses
