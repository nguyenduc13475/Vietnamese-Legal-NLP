import json
import os

# Định nghĩa các tập hợp nhãn hợp lệ
VALID_INTENTS = {"Obligation", "Prohibition", "Right", "Termination Condition", "Other"}
VALID_ENTITIES = {"PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"}


def validate_annotated_data(file_path="data/annotated_raw.json"):
    print(f"🔍 Đang đọc file: {file_path}")
    if not os.path.exists(file_path):
        print("❌ Lỗi: Không tìm thấy file. Hãy kiểm tra lại đường dẫn.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(
                f"❌ Lỗi: File JSON bị sai cú pháp (thiếu ngoặc, dư phẩy...). Chi tiết: {e}"
            )
            return

    print(f"⚡ Đang kiểm tra {len(data)} mệnh đề...\n")

    total_error_clauses = 0

    for index, item in enumerate(data):
        clause = item.get("clause", "")
        intent = item.get("intent", "")
        entities = item.get("entities", [])

        clause_errors = []

        # 1. Kiểm tra Ý định (Intent)
        if intent not in VALID_INTENTS:
            clause_errors.append(f"Intent không hợp lệ: '{intent}'")

        # 2. Kiểm tra Thực thể (Entities) theo thứ tự Trái -> Phải (Không lồng nhau)
        current_idx = 0  # Con trỏ vị trí hiện tại trong câu

        for ent in entities:
            text = ent.get("text", "")
            label = ent.get("label", "")

            # a. Label có hợp lệ không?
            if label not in VALID_ENTITIES:
                clause_errors.append(
                    f"Label không hợp lệ: '{label}' (thuộc text: '{text}')"
                )

            # b. Tìm text trong câu, bắt đầu từ vị trí con trỏ current_idx
            found_idx = clause.find(text, current_idx)

            if found_idx == -1:
                # Nếu không tìm thấy từ vị trí hiện tại trở đi (có thể do bị lồng nhau hoặc sai thứ tự)
                if text in clause:
                    clause_errors.append(
                        f"Text sai thứ tự, dư thừa, HOẶC bị lồng nhau trái phép: '{text}'"
                    )
                else:
                    clause_errors.append(f"Text bịa/không khớp nguyên văn: '{text}'")
            else:
                # c. Nếu tìm thấy, nhích con trỏ QUA HẲN cụm từ vừa tìm được (KHÔNG LỒNG NHAU)
                current_idx = found_idx + len(text)

        # Báo cáo lỗi nếu có
        if clause_errors:
            total_error_clauses += 1
            print(f"⚠️ Phát hiện lỗi ở Index thứ [{index}]:")
            print(f"   📝 Clause: {clause[:80]}...")
            for err in clause_errors:
                print(f"   ❌ {err}")
            print("-" * 50)

    # Tổng kết
    print("\n" + "=" * 50)
    if total_error_clauses == 0:
        print(
            "✅ HOÀN HẢO! File dữ liệu của bạn hoàn toàn hợp lệ (Chuẩn 1:1, Trái sang Phải, KHÔNG lồng nhau)."
        )
    else:
        print(f"🚨 TỔNG KẾT: Có {total_error_clauses} mệnh đề bị lỗi.")


if __name__ == "__main__":
    validate_annotated_data("data/annotated_raw.json")
