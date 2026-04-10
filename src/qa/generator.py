from langchain_core.prompts import PromptTemplate

from src.qa.retriever import LegalRetriever


class RAGGenerator:
    def __init__(self, llm_pipeline, retriever: LegalRetriever):
        """
        llm_pipeline: A Langchain compatible LLM (e.g., Google GenAI, OpenAI, or local HuggingFace pipeline)
        """
        self.llm = llm_pipeline
        self.retriever = retriever

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Bạn là một trợ lý pháp lý AI chuyên nghiệp. Dựa TUYỆT ĐỐI vào các điều khoản hợp đồng được cung cấp dưới đây, "
                "hãy trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và dễ hiểu.\n"
                "YÊU CẦU QUAN TRỌNG:\n"
                "1. CHỈ sử dụng thông tin từ 'Ngữ cảnh' (Context). KHÔNG ĐƯỢC TỰ BỊA RA CÂU TRẢ LỜI (Tuyệt đối không ảo giác).\n"
                "2. Nếu thông tin không có trong ngữ cảnh, hãy trả lời đúng nguyên văn: 'Tôi không tìm thấy thông tin liên quan trong hợp đồng.'\n"
                "3. Bắt buộc phải trích dẫn nguồn ở cuối câu trả lời theo định dạng: (Nguồn: [Trích lục nội dung mệnh đề gốc]).\n\n"
                "Ngữ cảnh:\n{context}\n\n"
                "Câu hỏi: {question}\n\n"
                "Trả lời:"
            ),
        )

    def ask(self, question: str) -> dict:
        # 1. Retrieve relevant clauses
        docs = self.retriever.retrieve(question)

        if not docs:
            return {
                "question": question,
                "answer": "Xin lỗi, tôi không tìm thấy thông tin nào trong hợp đồng liên quan đến câu hỏi của bạn.",
                "sources": [],
            }

        context = "\n".join([f"- {doc.page_content}" for doc in docs])

        # 2. Format Prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # 3. Generate Answer with Error Handling
        try:
            answer = self.llm.invoke(prompt)
            answer_text = answer.content if hasattr(answer, "content") else str(answer)
            answer_text = answer_text.strip()
        except Exception as e:
            answer_text = f"Language model call error (Please check GOOGLE_API_KEY in the .env file): {str(e)}"

        return {
            "question": question,
            "answer": answer_text,
            "sources": [doc.page_content for doc in docs],
        }
