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
            "2. MULTI-DOCUMENT HANDLING: If the user asks about multiple contracts, or if the context contains clauses from different files, synthesize the information clearly by separating the rules/conditions of each contract using bullet points or headers.\n"
            "3. If the information is not present in the context, answer exactly: 'I cannot find relevant information in the contract.'\n"
            "4. You must cite the source at the end of your answer in the format: (Source: [Extract of the original clause]).\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def ask(self, question: str, source_filter: str = None) -> dict:
        # 1. Retrieve relevant clauses from the Vector DB with optional filtering
        docs = self.retriever.retrieve(question, source_filter=source_filter)

        if not docs:
            return {
                "question": question,
                "answer": "Sorry, I could not find any information in the contract related to your question.",
                "sources": [],
            }

        context = "\n".join([f"- {doc.page_content}" for doc in docs])

        # 2. Format Prompt using native python strings
        prompt = self.prompt_template.format(context=context, question=question)

        # 3. Generate Answer using Google GenAI SDK
        if not self.client:
            answer_text = (
                "[CRITICAL ERROR: Google API Client not initialized. Check .env]"
            )
        else:
            try:
                # Set a low temperature for high factual accuracy in legal context
                response = self.client.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=1024
                    ),
                )
                answer_text = response.text.strip()
            except Exception as e:
                answer_text = f"LLM Generation Error: {str(e)}"

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ],
            "debug_prompt": prompt,  # Mandatory for teacher verification
        }
