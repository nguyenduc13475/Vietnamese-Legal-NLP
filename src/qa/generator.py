from google import genai
from google.genai import types

from src.qa.retriever import LegalRetriever


class RAGGenerator:
    def __init__(self, api_key: str, retriever: LegalRetriever):
        """
        Initialize RAG with the new google-genai SDK.
        """
        self.api_key = api_key
        self.client = None

        # Only initialize the client if a valid API key is provided
        if self.api_key and self.api_key != "your_google_gemini_api_key_here":
            self.client = genai.Client(api_key=self.api_key)

        self.retriever = retriever

        self.prompt_template = (
            "You are a professional legal AI assistant. Based STRICTLY on the contract clauses provided below, "
            "answer the user's question accurately, concisely, and comprehensively.\n"
            "IMPORTANT REQUIREMENTS:\n"
            "1. ONLY use information from the 'Context'. DO NOT HALLUCINATE OR MAKE UP ANSWERS.\n"
            "2. If the information is not present in the context, answer exactly: 'I cannot find relevant information in the contract.'\n"
            "3. You must cite the source at the end of your answer in the format: (Source: [Extract of the original clause]).\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def ask(self, question: str) -> dict:
        # 1. Retrieve relevant clauses from the Vector DB
        docs = self.retriever.retrieve(question)

        if not docs:
            return {
                "question": question,
                "answer": "Sorry, I could not find any information in the contract related to your question.",
                "sources": [],
            }

        context = "\n".join([f"- {doc.page_content}" for doc in docs])

        # 2. Format Prompt using native python strings
        prompt = self.prompt_template.format(context=context, question=question)

        # 3. Generate Answer with Error Handling using the new google-genai SDK
        if not self.client:
            answer_text = "[WARNING: GOOGLE_API_KEY not found in .env file]. Please provide a valid API Key to enable context-aware responses."
        else:
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",  # You can change this to gemini-2.5-flash if needed
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.1),
                )
                answer_text = response.text.strip()
            except Exception as e:
                answer_text = f"Language model call error: {str(e)}"

        return {
            "question": question,
            "answer": answer_text,
            "sources": [doc.page_content for doc in docs],
        }
