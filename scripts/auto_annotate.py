import json
import os
import random
import re
import sys
import time
from collections import Counter

from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.segmenter import segment_clauses

# Define label map to align BIO tags
TAG_MAP = {
    "O": 0,
    "B-PARTY": 1,
    "I-PARTY": 2,
    "B-MONEY": 3,
    "I-MONEY": 4,
    "B-DATE": 5,
    "I-DATE": 6,
    "B-RATE": 7,
    "I-RATE": 8,
    "B-PENALTY": 9,
    "I-PENALTY": 10,
    "B-LAW": 11,
    "I-LAW": 12,
}


def align_bio_tags(text: str, entities: list) -> tuple:
    """
    Tokenize the sentences and assign BIO tags to each token
    based on the entities identified by the LLM.
    """
    # Tokenize words and treat punctuation as separate tokens to
    # prevent them from being attached to the text.
    tokens = re.findall(r"\w+|[^\w\s]", text)
    tags = [0] * len(tokens)

    for ent in entities:
        ent_text = ent.get("text", "")
        ent_label = ent.get("label", "")
        if ent_label not in ["PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"]:
            continue

        ent_tokens = re.findall(r"\w+|[^\w\s]", ent_text)
        ent_len = len(ent_tokens)

        # Slide a window of length ent_len over the original tokens array to find a match.
        for i in range(len(tokens) - ent_len + 1):
            if tokens[i : i + ent_len] == ent_tokens:
                # Ensure it is not overwritten by another entity (prioritize the first match found)
                if all(tags[j] == 0 for j in range(i, i + ent_len)):
                    tags[i] = TAG_MAP[f"B-{ent_label}"]
                    for j in range(i + 1, i + ent_len):
                        tags[j] = TAG_MAP[f"I-{ent_label}"]
                break
    return tokens, tags


def extract_raw_clauses(input_dir: str, clauses_per_file: int = 50) -> list:
    """
    Extracts a fixed number of clauses per file to ensure a balanced
    starting point before data augmentation.
    """
    print(f"📖 Reading contracts from {input_dir}...")
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    selected_data = []
    total_clauses = 0

    for file in files:
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            text = f.read()
            clauses = segment_clauses(text)

            valid_clauses = []
            for c in clauses:
                c_clean = c.strip()
                has_letters = any(char.isalpha() for char in c_clean)
                word_count = len(c_clean.split())

                if has_letters and (
                    word_count >= 3
                    or c_clean.lower().startswith(("điều", "khoản", "mục", "phần"))
                ):
                    valid_clauses.append(c_clean)

            # Preserve order and remove duplicate clauses
            seen = set()
            unique_clauses = [
                x for x in valid_clauses if not (x in seen or seen.add(x))
            ]

            # Sample base clauses for this file
            sampled_clauses = random.sample(
                unique_clauses, min(clauses_per_file, len(unique_clauses))
            )

            selected_data.append(
                {"file": file, "full_text": text, "clauses": sampled_clauses}
            )
            total_clauses += len(sampled_clauses)

    print(f"Randomly selected {total_clauses} base clauses across {len(files)} files.")
    return selected_data


def call_gemini_json(
    client, prompt: str, max_retries: int = 3, temperature: float = 0.1
) -> list:
    """Helper function to cleanly call Gemini and parse JSON arrays."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            json_text = response.text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:-3].strip()

            return json.loads(json_text)

        except Exception as e:
            error_msg = str(e).upper()
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print("Rate limit reached. Sleeping for 60 seconds before retrying...")
                time.sleep(60)
            else:
                print(
                    f"API Error (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in 5 seconds..."
                )
                time.sleep(5)
    return []


def generate_annotations(selected_data: list):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        print("WARNING: GOOGLE_API_KEY is not set!")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    annotated_results = []

    # Minimum threshold per document to solve class imbalance
    BALANCE_TARGETS = {
        "intent": {
            "Prohibition": 4,
            "Termination Condition": 5,
            "Right": 6,
        },
        "entities": {"PENALTY": 6, "RATE": 6, "MONEY": 5, "LAW": 4},
    }

    ANNOTATE_PROMPT_TEMPLATE = """
    Bạn là chuyên gia pháp lý và ngôn ngữ học. Nhiệm vụ của bạn là phân tích các mệnh đề (clauses) được trích xuất từ một hợp đồng.
    Để giúp bạn hiểu rõ ngữ cảnh và tránh phân loại nhầm hoặc sinh ra nhãn sai, tôi sẽ cung cấp TOÀN BỘ NỘI DUNG hợp đồng. 
    Bạn chỉ cần phân tích và trả kết quả cho danh sách các mệnh đề được yêu cầu.

    --- NỘI DUNG HỢP ĐỒNG LÀM NGỮ CẢNH (Context) ---
    {full_text}
    ------------------------------------------------

    Đối với MỖI mệnh đề trong danh sách cần phân tích dưới đây, hãy:

    1. Phân loại Ý ĐỊNH (intent) BẮT BUỘC VÀO ĐÚNG 1 TRONG 5 LOẠI SAU:
       - Obligation (Nghĩa vụ phải làm, cam kết thực hiện, trách nhiệm thanh toán)
       - Prohibition (Bị cấm làm, không được phép, tuyệt đối không)
       - Right (Quyền lợi được hưởng, quyền yêu cầu, có quyền)
       - Termination Condition (Điều kiện chấm dứt, hủy bỏ hợp đồng, các trường hợp vi phạm bị phạt/bồi thường dẫn đến hậu quả)
       - Other (Các định nghĩa, thông tin chung, thời điểm có hiệu lực, tiêu đề)
       *Lưu ý: Các điều khoản phạt (Penalty) nên được xếp vào 'Obligation' hoặc 'Termination Condition'. Tuyệt đối không tạo nhãn mới.

    2. Trích xuất THỰC THỂ (entities) theo các nhãn sau (Chỉ trích xuất phần chữ có trong nguyên văn mệnh đề đang xét):
       - PARTY: Tên công ty, cá nhân, danh xưng các bên tham gia.
       - MONEY: Số tiền, giá trị tiền tệ cụ thể. (Lưu ý: Nếu số tiền là khoản tiền phạt vi phạm, hãy ưu tiên gán nhãn PENALTY thay vì MONEY).
       - DATE: Ngày, tháng, năm, thời hạn, mốc thời gian.
       - RATE: Tỷ lệ phần trăm hoặc hệ số (VD: 10%, 1.5 lần).
       - PENALTY: Mức phạt hoặc số tiền bồi thường cụ thể liên quan đến vi phạm (VD: phạt 8%, bồi thường 03 tháng tiền thuê, phạt 10.000.000 VNĐ).
       - LAW: Tên văn bản quy phạm pháp luật cụ thể. KHÔNG gán cho từ "pháp luật" chung chung.

    BẮT BUỘC TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON ARRAY CHO {count} MỆNH ĐỀ:
    [
      {{
        "clause": "<nguyên văn mệnh đề>",
        "intent": "<ý định>",
        "entities": [
          {{"text": "<phần chữ trích xuất chính xác 100% từ clause>", "label": "<nhãn>"}}
        ]
      }}
    ]

    --- DANH SÁCH MỆNH ĐỀ CẦN PHÂN TÍCH ---
    {clauses_json}
    """

    SYNTHETIC_PROMPT_TEMPLATE = """
    Bạn là chuyên gia pháp lý và tạo dữ liệu AI. 
    Dựa vào ngữ cảnh của hợp đồng dưới đây, hãy SÁNG TÁC (Generate) thêm một số mệnh đề mới hoàn toàn, nhưng phải giữ nguyên văn phong và logic của hợp đồng này.
    Mục đích là để tạo thêm dữ liệu huấn luyện AI cho những nhãn đang bị thiếu hụt.

    --- NGỮ CẢNH HỢP ĐỒNG ---
    {full_text}
    -------------------------

    YÊU CẦU SÁNG TÁC:
    Hãy tạo ra các câu có chứa các thông tin sau đây:
    {deficits_text}

    Hãy trộn lẫn các yêu cầu trên để tạo ra khoảng {total_sentences_needed} câu hợp lý.
    Quy tắc gán nhãn (intent và entities) y hệt như đã quy định chuẩn (Obligation, Prohibition, PENALTY, RATE...).

    BẮT BUỘC TRẢ VỀ ĐỊNH DẠNG JSON ARRAY CHO CÁC MỆNH ĐỀ VỪA SÁNG TÁC:
    [
      {{
        "clause": "<câu bạn vừa sáng tác>",
        "intent": "<ý định>",
        "entities": [
          {{"text": "<phần chữ trích xuất chính xác 100% từ câu sáng tác>", "label": "<nhãn>"}}
        ]
      }}
    ]
    """

    for data in selected_data:
        file_name = data["file"]
        full_text = data["full_text"]
        clauses = data["clauses"]

        if not clauses:
            continue

        print(f"\nProcessing {file_name}...")
        file_annotations = []

        # --- STEP 1: ANNOTATE EXISTING CLAUSES ---
        batch_size = 25
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i : i + batch_size]
            prompt = ANNOTATE_PROMPT_TEMPLATE.format(
                full_text=full_text,
                count=len(batch),
                clauses_json=json.dumps(batch, ensure_ascii=False, indent=2),
            )

            batch_data = call_gemini_json(client, prompt, temperature=0.1)
            file_annotations.extend(batch_data)
            print(
                f"  - Annotated batch {i // batch_size + 1} ({len(batch_data)}/{len(batch)} sentences)"
            )
            time.sleep(2)

        # --- STEP 2: CALCULATE IMBALANCE DEFICITS ---
        intent_counts = Counter(item.get("intent") for item in file_annotations)
        entity_counts = Counter(
            ent.get("label")
            for item in file_annotations
            for ent in item.get("entities", [])
        )

        deficits = []
        total_sentences_needed = 0

        for intent_name, target in BALANCE_TARGETS["intent"].items():
            if intent_counts[intent_name] < target:
                shortfall = target - intent_counts[intent_name]
                deficits.append(
                    f"- Cần thêm ít nhất {shortfall} câu mang ý định '{intent_name}'"
                )
                total_sentences_needed += shortfall

        for entity_name, target in BALANCE_TARGETS["entities"].items():
            if entity_counts[entity_name] < target:
                shortfall = target - entity_counts[entity_name]
                deficits.append(
                    f"- Cần thêm ít nhất {shortfall} câu chứa thực thể loại '{entity_name}'"
                )
                total_sentences_needed = max(
                    total_sentences_needed, shortfall
                )  # Overlap is fine

        # --- STEP 3: GENERATE SYNTHETIC CLAUSES TO FILL GAPS ---
        if deficits:
            print(
                f"Imbalance detected. Generating ~{total_sentences_needed} synthetic clauses to fill gaps..."
            )
            deficits_text = "\n".join(deficits)
            synthetic_prompt = SYNTHETIC_PROMPT_TEMPLATE.format(
                full_text=full_text,
                deficits_text=deficits_text,
                total_sentences_needed=total_sentences_needed + 2,  # Add a small buffer
            )

            synthetic_data = call_gemini_json(
                client, synthetic_prompt, temperature=0.3
            )  # Slightly higher temp for creativity
            file_annotations.extend(synthetic_data)
            print(
                f"  + Synthesized {len(synthetic_data)} missing clauses successfully."
            )
            time.sleep(2)
        else:
            print("Data is well balanced. No synthetic generation needed.")

        annotated_results.extend(file_annotations)

    return annotated_results


def split_and_save(annotated_data: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    intent_data = []
    ner_data = []

    for item in annotated_data:
        # Prepare data for the Intent model.
        intent_data.append(
            {"text": item.get("clause", ""), "label": item.get("intent", "Other")}
        )

        # Prepare data for the NER model.
        tokens, tags = align_bio_tags(item.get("clause", ""), item.get("entities", []))
        ner_data.append({"tokens": tokens, "ner_tags": tags})

    # Split 80/20
    split_idx = int(len(intent_data) * 0.8)

    datasets = {
        "intent_train.json": intent_data[:split_idx],
        "intent_test.json": intent_data[split_idx:],
        "ner_train.json": ner_data[:split_idx],
        "ner_test.json": ner_data[split_idx:],
    }

    for filename, data in datasets.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu {len(data)} dòng vào {filepath}")


if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    ANNOTATED_DIR = "data/annotated"

    # 1. Extract sentences and retrieve context from docs
    selected_data = extract_raw_clauses(PROCESSED_DIR)

    if selected_data:
        # 2. Call Gemini to annotate with full context.
        results = generate_annotations(selected_data)

        # 3. Split into Train/Test sets and format for PhoBERT, then export files.
        split_and_save(results, ANNOTATED_DIR)
        print("\nCompleted the Auto-Annotation process!")
    else:
        print("No text data found in the processed folder.")
