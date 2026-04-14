import json
import os
import sys

from transformers import AutoTokenizer

# Tải tokenizer của ViDeBERTa-xsmall
TOKENIZER_NAME = "Fsoft-AIC/videberta-xsmall"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
except Exception as e:
    print(f"Lỗi khi tải tokenizer {TOKENIZER_NAME}.\nChi tiết: {e}")
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
    Gán nhãn cho token dựa trên danh sách spans (tọa độ char) đã được chốt.
    """
    encoding = tokenizer(clause, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    tags = [default_label] * len(tokens)

    for span in spans:
        start_char = span["start"]
        end_char = span["end"]
        label = span["label"]

        is_first_token = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue

            # Token lies within the span
            if tok_start >= start_char and tok_end <= end_char:
                # Critical for Vietnamese: Check if token is purely whitespace
                if not tokens[i].strip():
                    tags[i] = "O"
                elif use_bio:
                    tags[i] = f"B-{label}" if is_first_token else f"I-{label}"
                    is_first_token = False
                else:
                    tags[i] = label

    return tokens, tags


def process_segmentation(segment_raw_path: str):
    """Xử lý dataset Segmentation"""
    print(f"📖 Đang xử lý Segmentation từ {segment_raw_path}...")
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
                components.append({"text": seg_text, "label": idx + 1})

        # Bước 1: Lấy tọa độ tuần tự
        spans = extract_spans_sequentially(clause, components)
        # Bước 2: Gán nhãn
        tokens, tags = assign_labels_to_tokens(clause, spans, default_label=0)

        dataset.append({"tokens": tokens, "segment_tags": tags})

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

        ner_data.append({"tokens": ner_tokens, "ner_tags": ner_tags})
        srl_data.append({"tokens": srl_tokens, "srl_tags": srl_tags})

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
