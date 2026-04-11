import argparse
import json
import os
import re
import sys

# Import segmenter từ source của bạn
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
    Tokenize các câu và gán nhãn BIO.
    MATCH 1:1 STRICT MODE - Tuân thủ tuyệt đối thứ tự và số lượng từ JSON.
    Mỗi object entity trong JSON chỉ khớp chính xác với 1 cụm token trong câu gốc.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    tags = [0] * len(tokens)

    # KHÔNG sắp xếp, KHÔNG gộp trùng lặp. Duyệt trực tiếp list do LLM trả về.
    for ent in entities:
        ent_text = ent.get("text", "")
        ent_label = ent.get("label", "")
        if ent_label not in ["PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"]:
            continue

        ent_tokens = re.findall(r"\w+|[^\w\s]", ent_text)
        ent_len = len(ent_tokens)
        if ent_len == 0:
            continue

        # Tìm vị trí đầu tiên trong câu khớp với cụm token và CHƯA BỊ CHIẾM
        for i in range(len(tokens) - ent_len + 1):
            if tokens[i : i + ent_len] == ent_tokens:
                # Kiểm tra lấp đầy (đảm bảo không đè lên nhãn của object đã khớp trước đó)
                if all(tags[j] == 0 for j in range(i, i + ent_len)):
                    tags[i] = TAG_MAP[f"B-{ent_label}"]
                    for j in range(i + 1, i + ent_len):
                        tags[j] = TAG_MAP[f"I-{ent_label}"]

                    # BẮT BUỘC CÓ BREAK: Đã khớp được 1 chỗ thì lập tức dừng,
                    # để lại các từ giống hệt (nếu có) phía sau cho các object JSON khác.
                    break

    return tokens, tags


def extract_raw_clauses(input_dir: str) -> list:
    """
    Sử dụng segment_clauses để cắt hợp đồng thành các câu,
    LẤY TOÀN BỘ danh sách mệnh đề hợp lệ theo đúng thứ tự.
    """
    print(f"📖 Đang đọc và cắt toàn bộ các hợp đồng từ {input_dir}...")
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    selected_data = []

    for file in files:
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            text = f.read()
            clauses = segment_clauses(text)

            valid_clauses = []
            for c in clauses:
                c_clean = c.strip()
                has_letters = any(char.isalpha() for char in c_clean)
                word_count = len(c_clean.split())

                # Lọc bỏ rác, giữ lại câu có ý nghĩa
                if has_letters and (
                    word_count >= 3
                    or c_clean.lower().startswith(("điều", "khoản", "mục", "phần"))
                ):
                    valid_clauses.append(c_clean)

            # Lọc trùng lặp nhưng vẫn giữ nguyên thứ tự gốc của hợp đồng
            seen = set()
            unique_clauses = [
                x for x in valid_clauses if not (x in seen or seen.add(x))
            ]

            selected_data.append({"file": file, "clauses": unique_clauses})

    return selected_data


def generate_prompts(selected_data: list, output_dir: str):
    """
    Tạo prompt yêu cầu AI TỰ DO CHỌN từ list full câu, sau đó gán nhãn và sinh thêm.
    """
    os.makedirs(output_dir, exist_ok=True)

    PROMPT_TEMPLATE = """Bạn là chuyên gia pháp lý và kỹ sư dữ liệu AI. 
Dưới đây là TOÀN BỘ danh sách các mệnh đề đã được HỆ THỐNG MÁY TÍNH CẮT TỰ ĐỘNG từ một hợp đồng, được xếp theo đúng thứ tự.
*Lưu ý: Do máy tính tự cắt nên một số câu có thể chứa các tiền tố lửng lơ như "Điều 1.", "Khoản 2.", số thứ tự. ĐÂY LÀ ĐIỀU CỐ Ý ĐỂ HUẤN LUYỆN MODEL.*

--- DANH SÁCH MỆNH ĐỀ GỐC CỦA HỢP ĐỒNG ---
{clauses_json}
------------------------------------------

NHIỆM VỤ CỦA BẠN:
1. LỰA CHỌN TỰ DO: Hãy tự do "cherry-pick" (lựa chọn) ra GẦN 50 MỆNH ĐỀ (nếu hợp đồng quá ngắn thì có thể chọn ít hơn) sao cho đa dạng và chứa nhiều thông tin pháp lý nhất (ưu tiên các câu có tiền phạt, ngày tháng, quyền/nghĩa vụ) từ danh sách trên để gán nhãn. TUYỆT ĐỐI GIỮ NGUYÊN VĂN 100% CÁC CÂU NÀY, không tự ý sửa đổi từ ngữ hay xóa bỏ "Điều...", "Khoản...".
2. XỬ LÝ IMBALANCE (SÁNG TÁC THÊM): Đánh giá sự thiếu hụt nhãn của gần 50 câu vừa chọn và SÁNG TÁC THÊM GẦN 25 MỆNH ĐỀ MỚI để cân bằng dữ liệu.
   -> *Quan trọng: Các câu sáng tác thêm CŨNG PHẢI MÔ PHỎNG LẠI CÁCH CẮT CÂU CỦA MÁY TÍNH (Thêm ngẫu nhiên "Điều X.", "Mục Y.", "a)" vào đầu câu để model làm quen).*
3. TỔNG KẾT: Trả về gần 75 MỆNH ĐỀ.

YÊU CẦU PHÂN LOẠI & GÁN NHÃN:
1. Ý ĐỊNH (intent) (Bắt buộc 1 trong 5 loại):
- Obligation (Nghĩa vụ phải làm, cam kết thực hiện, trách nhiệm thanh toán)
- Prohibition (Bị cấm làm, không được phép, tuyệt đối không)
- Right (Quyền lợi được hưởng, quyền yêu cầu, có quyền)
- Termination Condition (Điều kiện chấm dứt, hủy bỏ hợp đồng, các trường hợp vi phạm bị phạt/bồi thường dẫn đến hậu quả)
- Other (Các định nghĩa, thông tin chung, thời điểm có hiệu lực, tiêu đề)
*Lưu ý: Các điều khoản phạt (Penalty) nên được xếp vào 'Obligation' hoặc 'Termination Condition'. Tuyệt đối không tạo nhãn mới (ví dụ tạo thêm intent "Penalty" là sai).
   *CHÚ Ý: Đảm bảo 'Prohibition', 'Termination Condition', 'Right' có ÍT NHẤT 10 CÂU cho mỗi loại trong tổng số gần 75 câu.*

2. TRÍCH XUẤT THỰC THỂ (entities) - BẮT BUỘC TUÂN THỦ 3 QUY TẮC THÉP SAU:
   [Quy tắc 1: Quét từ Trái sang Phải] - Các thực thể trong JSON array phải được liệt kê theo ĐÚNG THỨ TỰ xuất hiện từ đầu câu đến cuối câu.
   [Quy tắc 2: Không lồng nhau] - Mỗi phần chữ (text) chỉ được thuộc về 1 nhãn duy nhất. TUYỆT ĐỐI KHÔNG trích xuất 2 thực thể đè lên nhau (VD: Không được xuất cả "phạt 8%" và "8%").
   [Quy tắc 3: Tính Nguyên Tử (Atomic)] - Chỉ lấy GIÁ TRỊ CỐT LÕI nhưng đủ Ý NGHĨA. Tuyệt đối KHÔNG gộp các động từ/quan hệ từ (như: "phạt", "bồi thường", "chịu", "bằng") vào trong thực thể.

DANH SÁCH NHÃN THỰC THỂ:
- PARTY: Tên công ty, cá nhân, danh xưng các bên tham gia (VD: Bên A, Bên B, Người Thuê, Người Cho Thuê).
- DATE: Ngày, tháng, năm, thời hạn, mốc thời gian (VD: 30 ngày, 15/08/2023).
- LAW: Tên văn bản quy phạm pháp luật cụ thể. KHÔNG gán cho từ "pháp luật" chung chung.
- PENALTY: (ƯU TIÊN CAO NHẤT). Bất kỳ chuỗi giá trị nào mang ý nghĩa PHẠT/BỒI THƯỜNG vi phạm thì CHỈ gán nhãn PENALTY. (VD: "8% giá trị phần nghĩa vụ bị vi phạm", "03 tháng tiền thuê", "10.000.000 VNĐ" --> TUYỆT ĐỐI KHÔNG trích xuất dài dòng kiểu "phạt 8% giá trị phần nghĩa vụ bị vi phạm" hay "bồi thường 10.000.000 VNĐ").
- MONEY: Số tiền, giá trị tiền tệ thông thường. (Luật loại trừ: Nếu số tiền đó là tiền phạt, TUYỆT ĐỐI KHÔNG dùng nhãn MONEY, chỉ dùng PENALTY).
- RATE: Tỷ lệ phần trăm hoặc hệ số thông thường (VD: 10%, 1.5 lần, 20%/Năm). Luật loại trừ: Nếu tỷ lệ đó là mức phạt, TUYỆT ĐỐI KHÔNG dùng nhãn RATE, chỉ dùng PENALTY. Ví dụ "10% giá trị cổ phần" thì "10%" là RATE.

   *CHÚ Ý XỬ LÝ IMBALANCE: Đảm bảo 'PENALTY', 'RATE', 'LAW' xuất hiện ÍT NHẤT 8-10 LẦN mỗi loại trong tổng số mệnh đề trả về.*

BẮT BUỘC TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON ARRAY SAU (Không sinh thêm text giải thích ngoài JSON):
[
  {{
    "clause": "<nguyên văn mệnh đề đã chọn hoặc mệnh đề sáng tác thêm>",
    "intent": "<ý định>",
    "entities": [
      {{"text": "<phần chữ trích xuất chính xác 100%>", "label": "<nhãn>"}}
    ]
  }}
]
"""

    count_files = 0
    for data in selected_data:
        file_name = data["file"]
        clauses = data["clauses"]

        if not clauses:
            continue

        clauses_json_str = json.dumps(clauses, ensure_ascii=False, indent=2)
        prompt_content = PROMPT_TEMPLATE.format(clauses_json=clauses_json_str)

        prompt_filename = file_name.replace(".txt", "_prompt.txt")
        with open(
            os.path.join(output_dir, prompt_filename), "w", encoding="utf-8"
        ) as f:
            f.write(prompt_content)
        count_files += 1

    print(f"✅ Đã tạo thành công {count_files} file prompt tại thư mục: {output_dir}")


def split_and_save(raw_json_path: str, output_dir: str):
    if not os.path.exists(raw_json_path):
        print(f"❌ Lỗi: Không tìm thấy file {raw_json_path}.")
        sys.exit(1)

    with open(raw_json_path, "r", encoding="utf-8") as f:
        annotated_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    intent_data, ner_data = [], []

    for item in annotated_data:
        clause_text = item.get("clause", "")
        if not clause_text:
            continue

        intent_data.append({"text": clause_text, "label": item.get("intent", "Other")})
        tokens, tags = align_bio_tags(clause_text, item.get("entities", []))
        ner_data.append({"tokens": tokens, "ner_tags": tags})

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
        print(f"  👉 Đã lưu {len(data)} mẫu vào {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Công cụ xử lý Hợp đồng thành Dataset (Off-API)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["generate", "parse"], required=True
    )
    args = parser.parse_args()

    PROCESSED_DIR = "data/processed"
    PROMPT_DIR = "data/prompt"
    RAW_JSON_PATH = "data/annotated_raw.json"
    ANNOTATED_DIR = "data/annotated"

    if args.mode == "generate":
        selected_data = extract_raw_clauses(PROCESSED_DIR)
        generate_prompts(selected_data, PROMPT_DIR)
        print(
            f"\n💡 Copy nội dung trong '{PROMPT_DIR}' lên Gemini Pro để lấy kết quả JSON."
        )

    elif args.mode == "parse":
        split_and_save(RAW_JSON_PATH, ANNOTATED_DIR)
