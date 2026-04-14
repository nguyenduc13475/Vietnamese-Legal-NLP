import json
import os
import sys

from transformers import AutoTokenizer

# Tải tokenizer của ViDeBERTa-xsmall
from underthesea import word_tokenize

TOKENIZER_NAME = "vinai/phobert-base"
try:
    # Use standard PhoBERT tokenizer (slow version usually more stable for syllable-to-word)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
except Exception as e:
    print(f"Error loading PhoBERT tokenizer: {e}")
    sys.exit(1)


def extract_spans_sequentially(clause: str, components: list) -> list:
    """
    Quét tuần tự từ trái sang phải để lấy tọa độ (start_char, end_char) chính xác.
    Đảm bảo việc tìm kiếm phụ thuộc vào thứ tự xuất hiện, tránh lỗi trùng text.
    """
    spans = []
    search_start_idx = 0

    for comp in components:
        substr = comp.get("text", "").strip()
        label = comp.get("label")

        if not substr:
            continue

        start_char = clause.find(substr, search_start_idx)

        # Fallback an toàn (phòng trường hợp list bị lệch khoảng trắng)
        if start_char == -1:
            start_char = clause.find(substr)
            if start_char == -1:
                continue

        end_char = start_char + len(substr)
        # Chốt vị trí con trỏ để quét phần tử tiếp theo không bị dính ngược lại
        search_start_idx = end_char

        spans.append(
            {"text": substr, "label": label, "start": start_char, "end": end_char}
        )

    return spans


def assign_labels_to_tokens(clause: str, spans: list, default_label, use_bio=False):
    """
    100% Correct logic for PhoBERT syllable-to-word alignment.
    Maps PhoBERT tokens back to original spans without string-search errors.
    """
    # 1. PhoBERT specific: Segment to handle word-level underscores
    # This is what PhoBERT was trained on
    segmented_clause = word_tokenize(clause, format="text")

    # 2. Tokenize and get tokens
    encoding = tokenizer(segmented_clause, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    # 3. Create a syllable-based reconstruction to map tokens to original character indices
    # We clean tokens of PhoBERT specific artifacts ('@@' and '_')
    tags = [default_label] * len(tokens)

    # We rebuild the string from tokens to find exactly where they hit the original clause
    # Use a relative pointer that ignores whitespace and underscores discrepancy
    clause_stripped = "".join(clause.split()).lower()
    token_spans_in_stripped = []
    current_pos = 0

    for token in tokens:
        clean_tok = token.replace("@@", "").replace("_", "").lower()
        if not clean_tok:
            token_spans_in_stripped.append((-1, -1))
            continue

        start = clause_stripped.find(clean_tok, current_pos)
        if start != -1:
            end = start + len(clean_tok)
            token_spans_in_stripped.append((start, end))
            current_pos = end
        else:
            token_spans_in_stripped.append((-1, -1))

    # 4. Map the original character spans to the 'stripped' space
    # This allows us to compare apples to apples
    for span in spans:
        # Calculate start/end in whitespace-stripped version of original clause
        # This makes the span index match our token pointer
        prefix = clause[: span["start"]]
        prefix_stripped_len = len("".join(prefix.split()))
        content_stripped_len = len("".join(span["text"].split()))

        stripped_start = prefix_stripped_len
        stripped_end = stripped_start + content_stripped_len

        label = span["label"]
        is_first_token_in_span = True

        for i, (tok_s, tok_e) in enumerate(token_spans_in_stripped):
            if tok_s == -1:
                continue

            # Intersection check: if token is inside the stripped span
            if tok_s >= stripped_start and tok_e <= stripped_end:
                if use_bio:
                    tags[i] = f"B-{label}" if is_first_token_in_span else f"I-{label}"
                    is_first_token_in_span = False
                else:
                    tags[i] = label

    # 5. BIO Consistency Post-Process: Fix any O -> I-Tag errors
    if use_bio:
        for i in range(1, len(tags)):
            if tags[i].startswith("I-"):
                prev_tag = tags[i - 1]
                target_label = tags[i].split("-")[1]
                # If prev tag is O or a different label, change I- to B-
                if prev_tag == "O" or (
                    prev_tag != f"B-{target_label}" and prev_tag != f"I-{target_label}"
                ):
                    tags[i] = f"B-{target_label}"

    return tokens, tags


def process_segmentation(segment_raw_path: str):
    """Xử lý dataset Segmentation using the same ID-locking logic as NER"""
    print(f"📖 Processing Segmentation from {segment_raw_path}...")
    with open(segment_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = []
    for item in raw_data:
        clause = item.get("clause", "")
        segments = item.get("segment", [])
        if not clause or not segments:
            continue

        components = []
        for idx, seg_text in enumerate(segments):
            if seg_text.strip():
                # Label is the index + 1 (1, 2, 3, 4, 5)
                components.append({"text": seg_text, "label": idx + 1})

        # Use the EXACT same coordinate-transformation logic as NER
        spans = extract_spans_sequentially(clause, components)

        # assign_labels_to_tokens is our 'bulletproof' math function
        tokens, tags = assign_labels_to_tokens(
            clause, spans, default_label=0, use_bio=False
        )

        # Convert tokens to IDs immediately to lock the dataset
        token_ids = [
            tokenizer.convert_tokens_to_ids(t)
            if tokenizer.convert_tokens_to_ids(t) is not None
            else tokenizer.unk_token_id
            for t in tokens
        ]

        # Now Segment data looks exactly like NER data
        dataset.append({"input_ids": token_ids, "segment_tags": tags})

    return dataset


def process_annotated_tasks(annotated_raw_path: str):
    """Xử lý dataset Intent, NER và SRL"""
    print(f"📖 Đang xử lý Intent, NER, SRL từ {annotated_raw_path}...")
    with open(annotated_raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    intent_data, ner_data, srl_data = [], [], []

    for item in raw_data:
        clause = item.get("clause", "")
        if not clause:
            continue

        # --- 1. INTENT ---
        intent_data.append({"text": clause, "label": item.get("intent", "Other")})

        # --- 2. BÓC TÁCH TỌA ĐỘ SRL & NER ĐỘC LẬP ---
        ultra_ner = item.get("ultra_ner", [])
        srl_roles = item.get("srl_roles", [])

        srl_spans = extract_spans_sequentially(clause, srl_roles)
        ner_spans = extract_spans_sequentially(clause, ultra_ner)

        # --- 3. EMBED PREDICATE VÀO NER & XOÁ KHỎI SRL ---
        # Lấy PREDICATE ra để dùng cho NER
        predicate_spans = [span for span in srl_spans if span["label"] == "PREDICATE"]

        # CHÍNH SỬA TẠI ĐÂY: Loại bỏ PREDICATE khỏi danh sách SRL trước khi gán nhãn
        srl_spans = [span for span in srl_spans if span["label"] != "PREDICATE"]

        # Nhét vào NER và sort lại
        combined_ner_spans = ner_spans + predicate_spans
        combined_ner_spans.sort(key=lambda x: x["start"])

        # --- 4. GÁN NHÃN CHO TOKENS ---
        ner_tokens, ner_tags = assign_labels_to_tokens(
            clause, combined_ner_spans, default_label="O", use_bio=True
        )
        # Lúc này srl_tags sẽ không còn nhãn PREDICATE nữa
        srl_tokens, srl_tags = assign_labels_to_tokens(
            clause, srl_spans, default_label="OTHER", use_bio=False
        )

        # Convert tokens to IDs immediately during annotation to lock in the correct vocab mapping
        token_ids = [
            tokenizer.convert_tokens_to_ids(t)
            if tokenizer.convert_tokens_to_ids(t) is not None
            else tokenizer.unk_token_id
            for t in ner_tokens
        ]

        ner_data.append({"input_ids": token_ids, "ner_tags": ner_tags})
        srl_data.append({"input_ids": token_ids, "srl_tags": srl_tags})

    return intent_data, ner_data, srl_data


def split_and_save_all(segment_raw_path: str, annotated_raw_path: str, output_dir: str):
    if not os.path.exists(segment_raw_path):
        print(f"Lỗi: Không tìm thấy file {segment_raw_path}")
        sys.exit(1)
    if not os.path.exists(annotated_raw_path):
        print(f"Lỗi: Không tìm thấy file {annotated_raw_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    segmentation_data = process_segmentation(segment_raw_path)
    intent_data, ner_data, srl_data = process_annotated_tasks(annotated_raw_path)

    # Chia dataset 90/10
    datasets = {
        "segment_train.json": segmentation_data[: int(len(segmentation_data) * 0.9)],
        "segment_test.json": segmentation_data[int(len(segmentation_data) * 0.9) :],
        "intent_train.json": intent_data[: int(len(intent_data) * 0.9)],
        "intent_test.json": intent_data[int(len(intent_data) * 0.9) :],
        "ner_train.json": ner_data[: int(len(ner_data) * 0.9)],
        "ner_test.json": ner_data[int(len(ner_data) * 0.9) :],
        "srl_train.json": srl_data[: int(len(srl_data) * 0.9)],
        "srl_test.json": srl_data[int(len(srl_data) * 0.9) :],
    }

    for filename, data in datasets.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f" 👉 Đã lưu {len(data):>5} mẫu vào {filepath}")


if __name__ == "__main__":
    SEGMENT_RAW_JSON = "data/segment_raw.json"
    ANNOTATED_RAW_JSON = "data/annotated_raw.json"
    OUTPUT_DIRECTORY = "data/annotated"

    split_and_save_all(SEGMENT_RAW_JSON, ANNOTATED_RAW_JSON, OUTPUT_DIRECTORY)
    print("\n✅ Hoàn tất quá trình tạo dataset với logic tuần tự SRL -> Embed NER!")
