import argparse
import os
import time

import aspose.words as aw
from dotenv import load_dotenv
from google import genai
from google.genai import types


def batch_clean(input_dir: str, output_dir: str):
    """
    Use Gemini API to clean and anonymize contract templates.
    Send raw files directly to Gemini for analysis instead of manually extracting text.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if not f.startswith(".")]
    print(f"Found {len(files)} files to process in {input_dir}\n")

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        print("WARNING: GOOGLE_API_KEY is not validly set in .env!")
        return

    client = genai.Client(api_key=api_key)

    success_count = 0
    prompt = (
        "Bạn là một chuyên gia pháp lý và xử lý ngôn ngữ tự nhiên. "
        "Hãy đọc toàn bộ tài liệu này, làm sạch và CẤU TRÚC LẠI nội dung của nó. Yêu cầu BẮT BUỘC:\n"
        "1. QUY TẮC TÁCH DÒNG (QUAN TRỌNG NHẤT): MỖI DÒNG TRẢ VỀ PHẢI CHỨA ĐÚNG MỘT MỆNH ĐỀ / CÂU ĐỘC LẬP VỀ NGỮ NGHĨA. Nếu một Điều/Khoản có nhiều câu hoặc nhiều mục liệt kê, HÃY XUỐNG DÒNG cho mỗi câu/mục đó. Sao cho mỗi câu có thể phân tích về mặt ngữ pháp một cách hoàn chỉnh.\n"
        "2. Xóa bỏ số trang, tiêu đề lặp lại, mục lục, ký tự rác, và các dòng chữ đánh dấu bản quyền (watermark) của phần mềm chuyển đổi nếu có.\n"
        "3. NẾU LÀ MẪU TRỐNG (chứa dấu '...', '___', '[Tên]'): BẮT BUỘC BỊA RA THÔNG TIN THỰC TẾ (tên người, ngày tháng, số tiền, địa chỉ, tên công ty...) để điền vào TẤT CẢ các chỗ trống. Không để lại bất kỳ chỗ cần điền nào.\n"
        "4. Giữ nguyên cấu trúc các Điều, Khoản, Mục.\n"
        "5. KHÔNG giải thích, KHÔNG thêm lời chào, CHỈ trả về toàn bộ văn bản hợp đồng đã được làm sạch và điền đủ thông tin dạng Plain Text."
        '6. PHẢI dùng dấu " cho mọi kiểu dấu nháy kép.'
        "7. ĐẶC BIỆT QUAN TRỌNG: Cuối mỗi câu hoặc cuối mỗi mục liệt kê (ví dụ: a, b, c, d...) BẮT BUỘC PHẢI thêm dấu chấm (.). Không được để lửng câu để module Sentence Tokenizer có thể cắt mệnh đề chính xác."
        "8. LÁCH LUẬT BẢN QUYỀN: Hãy paraphrase (diễn đạt lại bằng từ đồng nghĩa) hoặc thay đổi một chút xíu cấu trúc câu văn của toàn bộ hợp đồng. TUYỆT ĐỐI KHÔNG trả về kết quả giống y hệt 100% văn bản gốc để tránh lỗi Recitation của hệ thống, nhưng vẫn phải giữ đúng 100% ý nghĩa pháp lý."
        "9. Trường hợp là hợp đồng song ngữ, không được gộp chung câu tiếng Việt vào câu Tiếng Anh tương ứng, phải tách làm 2 câu riêng biệt."
    )

    for filename in files:
        # Pre-calculate the output file path for verification.
        out_filename = os.path.splitext(filename)[0] + ".txt"
        out_filepath = os.path.join(output_dir, out_filename)

        # CHECK: SKIP IF FILE ALREADY EXISTS
        if os.path.exists(out_filepath):
            print(f"⏭️ Bỏ qua {filename}: Đã có file đã xử lý tại {out_filepath}")
            continue

        filepath = os.path.join(input_dir, filename)
        upload_filepath = filepath
        temp_pdf_path = None
        uploaded_file = None

        if not filename.lower().endswith(".pdf"):
            try:
                print(f"Converting to PDF: {filename}")
                doc = aw.Document(filepath)
                temp_pdf_path = os.path.splitext(filepath)[0] + "_temp.pdf"
                doc.save(temp_pdf_path)
                upload_filepath = temp_pdf_path
            except Exception as e:
                print(f"Error converting {filename} to PDF: {str(e)}")
                continue

        print(f"Uploading file to Gemini: {os.path.basename(upload_filepath)}")

        try:
            max_retries = 5

            for attempt in range(max_retries):
                try:
                    # Sending the raw file directly to Gemini (Upload only if not already uploaded)
                    if not uploaded_file:
                        uploaded_file = client.files.upload(file=upload_filepath)

                    response = client.models.generate_content(
                        model="gemini-3.1-flash-lite-preview",
                        contents=[prompt, uploaded_file],
                        config=types.GenerateContentConfig(temperature=0.4),
                    )
                    cleaned_text = response.text.strip()

                    with open(out_filepath, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)

                    print(f"Clean file saved: {out_filepath}")
                    success_count += 1
                    break

                except Exception as api_err:
                    if attempt < max_retries - 1:
                        wait_time = 6  # Thay đổi: Chờ cố định 60 giây (1 phút)
                        print(
                            f"API Overload/Error. Waiting {wait_time}s before retrying (Attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise api_err

            time.sleep(2)

        except Exception as e:
            print(
                f"API processing error for {filename} (attempted {max_retries} times): {str(e)}"
            )

        finally:
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except Exception as e:
                    print(f"Unable to delete temporary files {temp_pdf_path}: {e}")

    print(f"\nDone! Successfully cleaned {success_count} new contracts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean raw documents using Gemini API direct file upload"
    )
    parser.add_argument(
        "--input", default="data/raw", help="Directory containing raw contract files"
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output folder for cleaned text files",
    )
    args = parser.parse_args()

    batch_clean(args.input, args.output)
