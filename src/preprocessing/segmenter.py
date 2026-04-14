import re

import torch
from transformers import AutoTokenizer, pipeline

MODEL_PATH = "models/fine_tuned_segmenter"
_segment_pipeline = None


def get_segment_pipeline():
    """Lazy load the DL Segmenter to prevent startup crashes."""
    global _segment_pipeline
    if _segment_pipeline is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("Fsoft-AIC/videberta-xsmall")
            _segment_pipeline = pipeline(
                "token-classification",
                model=MODEL_PATH,
                tokenizer=tokenizer,
                aggregation_strategy=None,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            print(f"Warning: Could not load DL Segmenter. Using fallback. Error: {e}")
            _segment_pipeline = "fallback"
    return _segment_pipeline


def segment_clauses(text: str) -> list[dict]:
    """
    Extracts independent clauses line-by-line.
    Extracts context tags and ALIASES for RAG metadata.
    Uses Deep Learning (ViDeBERTa) to split composite clauses (Dạng 1 & Dạng 2).
    """
    raw_lines = text.split("\n")
    clauses = []
    current_context = "General"
    current_aliases = "[]"  # Default empty list string

    # Regex to identify basic list items (1., a), I., -)
    bullet_pattern = re.compile(
        r"^([a-zđ][\)\.]|\d+(?:\.\d+)*[\.\):]?|[IVXLCDM]+[\.\)]|[-+])\s+", re.IGNORECASE
    )

    seg_pipe = get_segment_pipeline()

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Detect ALIASES line - skip adding as clause but store for metadata
        if line.startswith("[ALIASES]"):
            current_aliases = line.replace("[ALIASES]", "", 1).strip()
            continue

        # Detect and extract the representative Title marker
        is_title = False
        if line.startswith("[TITLE]"):
            is_title = True
            line = line.replace("[TITLE]", "", 1).strip()

        # Extract context if present (e.g., "[Điều 1] Nội dung...")
        ctx_match = re.match(r"^\[(.*?)\]\s*(.*)", line)
        if ctx_match:
            current_context = ctx_match.group(1).strip()
            line = ctx_match.group(2).strip()

        # Clean up bullet points at the start
        clean_line = bullet_pattern.sub("", line).strip()

        # Discard fragments that are too short to be meaningful
        if len(clean_line) <= 5:
            continue

        # --- Deep Learning Composite Clause Splitting ---
        if seg_pipe and seg_pipe != "fallback":
            try:
                preds = seg_pipe(clean_line)
                blocks = {i: [] for i in range(1, 6)}

                for p in preds:
                    # Extract label integer (assuming format like 'LABEL_1', 'LABEL_2')
                    lbl_str = p["entity"].split("_")[-1]
                    if lbl_str.isdigit():
                        lbl = int(lbl_str)
                        if lbl in blocks:
                            # Clean up RoBERTa/DeBERTa special token markers (U+2581 and Ġ)
                            word = (
                                p["word"]
                                .replace("\u2581", " ")
                                .replace("Ġ", " ")
                                .strip()
                            )
                            blocks[lbl].append(word)

                # Reconstruct text for each block
                b = {
                    k: " ".join(v).replace(" ,", ",").replace(" .", ".").strip()
                    for k, v in blocks.items()
                }

                if any(b.values()):
                    has_5 = len(b[5]) > 0
                    if has_5:
                        # Type 2: A B C D E -> A B C, A D C E
                        c1 = f"{b[1]} {b[2]} {b[3]}".strip()
                        c2 = f"{b[1]} {b[4]} {b[3]} {b[5]}".strip()
                        if c1:
                            clauses.append(
                                {
                                    "text": c1,
                                    "context": current_context,
                                    "is_title": is_title,
                                    "aliases": current_aliases,
                                }
                            )
                        if c2:
                            clauses.append(
                                {
                                    "text": c2,
                                    "context": current_context,
                                    "is_title": False,
                                    "aliases": current_aliases,
                                }
                            )
                    else:
                        # Type 1: A B C D -> A B D, A C D
                        c1 = f"{b[1]} {b[2]} {b[4]}".strip()
                        c2 = f"{b[1]} {b[3]} {b[4]}".strip()
                        if c1:
                            clauses.append(
                                {
                                    "text": c1,
                                    "context": current_context,
                                    "is_title": is_title,
                                    "aliases": current_aliases,
                                }
                            )
                        if c2:
                            clauses.append(
                                {
                                    "text": c2,
                                    "context": current_context,
                                    "is_title": False,
                                    "aliases": current_aliases,
                                }
                            )

                    continue  # Skip the default append if DL handled it successfully
            except Exception as e:
                print(f"DL Segmentation error on: {clean_line[:30]}... Error: {e}")
                # Fallthrough to default append on error

        # --- Default fallback (No splitting) ---
        clauses.append(
            {
                "text": clean_line,
                "context": current_context,
                "is_title": is_title,
                "aliases": current_aliases,
            }
        )

    return clauses
