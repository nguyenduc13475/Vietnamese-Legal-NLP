import json
import os


def validate_dataset(data):
    all_valid = True

    for index, item in enumerate(data):
        # 1. Kiểm tra các key bắt buộc
        if not all(key in item for key in ("clause", "type", "segment")):
            print(
                f"[Lỗi - Item {index}] Thiếu một trong các key: 'clause', 'type', 'segment'."
            )
            all_valid = False
            continue

        clause = item.get("clause")
        item_type = item.get("type")
        segment = item.get("segment")

        # 2. Kiểm tra kiểu dữ liệu của clause và type
        if not isinstance(clause, str):
            print(f"[Lỗi - Item {index}] 'clause' phải là kiểu string.")
            all_valid = False

        if item_type not in [1, 2]:
            print(
                f"[Lỗi - Item {index}] 'type' phải là 1 hoặc 2 (hiện tại là {item_type})."
            )
            all_valid = False

        # 3. Kiểm tra kiểu dữ liệu và số lượng phần tử của segment
        if not isinstance(segment, list) or not all(
            isinstance(s, str) for s in segment
        ):
            print(f"[Lỗi - Item {index}] 'segment' phải là một list chứa các string.")
            all_valid = False
            continue  # Bỏ qua check tuần tự nếu segment không hợp lệ

        if item_type == 1 and len(segment) != 4:
            print(
                f"[Lỗi - Item {index}] type = 1 nhưng segment có {len(segment)} phần tử (yêu cầu 4)."
            )
            all_valid = False
        elif item_type == 2 and len(segment) != 5:
            print(
                f"[Lỗi - Item {index}] type = 2 nhưng segment có {len(segment)} phần tử (yêu cầu 5)."
            )
            all_valid = False

        # 4. Kiểm tra tính tuần tự của các chuỗi con trong clause
        current_pointer = 0
        for seg_idx, part in enumerate(segment):
            if part == "":
                continue  # Bỏ qua các chuỗi rỗng

            # Tìm vị trí xuất hiện của 'part' bắt đầu từ current_pointer
            found_idx = clause.find(part, current_pointer)

            if found_idx == -1:
                print(
                    f"[Lỗi - Item {index}] Chuỗi '{part}' (tại segment[{seg_idx}]) không xuất hiện tuần tự trong clause."
                )
                print(f"  => Clause gốc: {clause}")
                print(f"  => Bắt đầu tìm từ vị trí: {current_pointer}")
                all_valid = False
                break

            # Di chuyển con trỏ đến ngay sau chuỗi vừa tìm được
            current_pointer = found_idx + len(part)

    if all_valid:
        print("✅ Tuyệt vời! Toàn bộ dữ liệu đều hợp lệ.")
    else:
        print("❌ Dữ liệu có lỗi. Vui lòng kiểm tra lại log bên trên.")

    return all_valid


# ==========================================
# CÁCH CHẠY VỚI FILE JSON
# ==========================================
if __name__ == "__main__":
    # Tên file JSON của bạn (đặt cùng thư mục với script này)
    file_path = "data/segment_raw.json"

    if not os.path.exists(file_path):
        print(
            f"Không tìm thấy file '{file_path}'. Vui lòng đảm bảo file tồn tại ở cùng thư mục."
        )
    else:
        print(f"Đang đọc dữ liệu từ '{file_path}' và kiểm tra...\n")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            # Gọi hàm validate
            validate_dataset(dataset)

        except json.JSONDecodeError as e:
            print(f"Lỗi cú pháp JSON trong file: {e}")
        except Exception as e:
            print(f"Đã xảy ra lỗi không xác định: {e}")
