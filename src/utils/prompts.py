# Standardized prompt for cleaning and formatting raw legal contracts via Gemini.
# This ensures structural consistency between offline batch processing and real-time API ingestion.
DOCUMENT_CLEANING_PROMPT = (
    "Bạn là một chuyên gia pháp lý và xử lý ngôn ngữ tự nhiên. "
    "Hãy đọc toàn bộ tài liệu này, làm sạch và CẤU TRÚC LẠI nội dung của nó. Yêu cầu BẮT BUỘC:\n"
    "1. QUY TẮC TÁCH DÒNG (QUAN TRỌNG NHẤT): MỖI DÒNG TRẢ VỀ PHẢI CHỨA ĐÚNG MỘT MỆNH ĐỀ / CÂU ĐỘC LẬP VỀ NGỮ NGHĨA. Nếu một Điều/Khoản có nhiều câu hoặc nhiều mục liệt kê, HÃY XUỐNG DÒNG cho mỗi câu/mục đó. Sao cho mỗi câu có thể phân tích về mặt ngữ pháp một cách hoàn chỉnh.\n"
    "2. Xóa bỏ số trang, tiêu đề lặp lại, mục lục, ký tự rác, và các dòng chữ đánh dấu bản quyền (watermark) của phần mềm chuyển đổi nếu có.\n"
    "3. NẾU LÀ MẪU TRỐNG (chứa dấu '...', '___', '[Tên]'): BẮT BUỘC BỊA RA THÔNG TIN THỰC TẾ (tên người, ngày tháng, số tiền, địa chỉ, tên công ty...) để điền vào TẤT CẢ các chỗ trống. Không để lại bất kỳ chỗ cần điền nào.\n"
    "4. Giữ nguyên cấu trúc các Điều, Khoản, Mục.\n"
    "5. KHÔNG giải thích, KHÔNG thêm lời chào, CHỈ trả về toàn bộ văn bản hợp đồng đã được làm sạch và điền đủ thông tin dạng Plain Text.\n"
    '6. PHẢI dùng dấu " cho mọi kiểu dấu nháy kép.\n'
    "7. ĐẶC BIỆT QUAN TRỌNG: Cuối mỗi câu hoặc cuối mỗi mục liệt kê (ví dụ: a, b, c, d...) BẮT BUỘC PHẢI thêm dấu chấm (.). Không được để lửng câu để module Sentence Tokenizer có thể cắt mệnh đề chính xác.\n"
    "8. LÁCH LUẬT BẢN QUYỀN: Hãy paraphrase (diễn đạt lại bằng từ đồng nghĩa) hoặc thay đổi một chút xíu cấu trúc câu văn của toàn bộ hợp đồng. TUYỆT ĐỐI KHÔNG trả về kết quả giống y hệt 100% văn bản gốc để tránh lỗi Recitation của hệ thống, nhưng vẫn phải giữ đúng 100% ý nghĩa pháp lý.\n"
    "9. Trường hợp là hợp đồng song ngữ, không được gộp chung câu tiếng Việt vào câu Tiếng Anh tương ứng, phải tách làm 2 câu riêng biệt."
    "10. GẮN THẺ NGỮ CẢNH (BẮT BUỘC): Phân tích cấu trúc phân cấp của hợp đồng (Phần, Chương, Điều, Khoản, Điểm, ...). Trước MỖI dòng/câu trả về, bạn BẮT BUỘC phải gắn thẻ ngữ cảnh hiện tại trong dấu ngoặc vuông `[]` ngoại trừ trường hợp câu sau có cùng ngữ cảnh với câu trước thì không cần. Ví dụ: `[Điều 1 - Khoản 2] Bên A sẽ thanh toán...`. Nếu là phần mở đầu/chữ ký, ghi `[Thông tin chung]` hoặc `[Phần ký tên]`.\n"
    "11. ĐÁNH DẤU TIÊU ĐỀ HỢP ĐỒNG (QUAN TRỌNG): Tìm đúng 1 câu chứa Tên/Tiêu đề chính của hợp đồng và thêm tiền tố `[TITLE]` vào ngay đầu dòng đó. Ví dụ: `[TITLE] [Thông tin chung] HỢP ĐỒNG CHO THUÊ LẠI`.\n"
)
