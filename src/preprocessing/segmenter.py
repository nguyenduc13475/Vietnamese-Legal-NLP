import re

import torch
from transformers import AutoTokenizer, pipeline

MODEL_PATH = "models/fine_tuned_segmenter"
_segment_pipeline = None


def get_segment_pipeline():
    """Lazy load the DL Segmenter to handle Type 1 & 2 splitting."""
    global _segment_pipeline
    if _segment_pipeline is None:
        try:
            # Using fast tokenizer for better performance
            tokenizer = AutoTokenizer.from_pretrained(
                "Fsoft-AIC/videberta-xsmall", use_fast=True
            )
            _segment_pipeline = pipeline(
                "token-classification",
                model=MODEL_PATH,
                tokenizer=tokenizer,
                aggregation_strategy=None,  # We need raw tokens to handle non-continuous blocks
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"DL Segmenter load failed: {e}. Falling back to line-based.")
            _segment_pipeline = "fallback"
    return _segment_pipeline


def segment_clauses(text: str) -> list[dict]:
    """
    Splits text into clauses. Uses ViDeBERTa to identify components A,B,C,D,E
    and reconstructs them into valid legal sentences.
    """
    raw_lines = text.split("\n")
    clauses = []
    current_context = "General"
    current_aliases = "[]"

    seg_pipe = get_segment_pipeline()
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

        if seg_pipe and seg_pipe != "fallback":
            try:
                preds = seg_pipe(clean_line)
                # Logic: collect tokens for each label (1-5), even if non-continuous
                blocks = {i: [] for i in range(1, 6)}
                for p in preds:
                    lbl_str = p["entity"].split("_")[-1]
                    if lbl_str.isdigit():
                        lbl = int(lbl_str)
                        if lbl in blocks:
                            word = p["word"].replace(" ", " ").strip()
                            blocks[lbl].append(word)

                b = {k: " ".join(v).strip() for k, v in blocks.items()}

                # Type detection and reconstruction
                if b[1] and b[2] and b[3]:
                    if b[5]:  # Type 2: A B C D E -> ABC + ADCE
                        c_list = [
                            f"{b[1]} {b[2]} {b[3]}",
                            f"{b[1]} {b[4]} {b[3]} {b[5]}",
                        ]
                    elif b[4]:  # Type 1: A B C D -> ABD + ACD
                        c_list = [f"{b[1]} {b[2]} {b[4]}", f"{b[1]} {b[3]} {b[4]}"]
                    else:
                        c_list = [clean_line]

                    for idx, c_text in enumerate(c_list):
                        clauses.append(
                            {
                                "text": c_text.strip(),
                                "context": current_context,
                                "is_title": is_title if idx == 0 else False,
                                "aliases": current_aliases,
                            }
                        )
                    continue
            except Exception as e:
                print(f"Segmentation logic error: {e}")

        # Fallback
        clauses.append(
            {
                "text": clean_line,
                "context": current_context,
                "is_title": is_title,
                "aliases": current_aliases,
            }
        )

    return clauses
