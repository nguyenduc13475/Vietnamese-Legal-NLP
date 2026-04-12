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
            "You are a professional legal AI assistant. Based STRICTLY on the contract context provided below, "
            "answer the user's question in Vietnamese. "
            "IMPORTANT REQUIREMENTS:\n"
            "1. CITATION STYLE: For every fact or rule mentioned, you MUST explicitly state the source and section using the format: 'Căn cứ theo Tên hợp đồng - Điều/Khoản/Điểm/..., blah blah ...'.\n"
            "2. MULTI-DOCUMENT QUERY: If the question involves different contracts, explicitly contrast the terms of each contract by name. If a contract mentioned in the question is missing from the provided context, state that clearly.\n"
            "3. ONLY use the provided context. If the answer is not there, say: 'I cannot find relevant information in the contract.'\n"
            "4. Accuracy is paramount. Use formal legal Vietnamese terminology.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def ask(self, question: str, source_filter: str = None) -> dict:
        # Phase 1: Pure Vector-Based Query Routing using [TITLE] tags
        target_sources = None

        if source_filter and source_filter not in ["All Contracts", "Tất cả Hợp đồng"]:
            # User manually targeted a specific file
            target_sources = [source_filter]
        else:
            # User selected 'All Contracts'. Execute Phase 1 search against Title vectors.
            # We fetch top 2 most relevant documents to handle comparison questions naturally.
            found_sources = self.retriever.get_top_sources_by_title(question, top_k=2)

            if found_sources:
                target_sources = found_sources
                print(f"Phase 1 Routing: Narrowed search to {target_sources}")
            else:
                print(
                    "Phase 1 Routing: No titles matched. Defaulting to global search."
                )

        # Phase 2: Targeted Retrieval from the Vector DB
        docs = self.retriever.retrieve(
            question, top_k=10, source_filters=target_sources
        )

        if not docs:
            return {
                "question": question,
                "answer": "Sorry, I could not find any information in the contract related to your question.",
                "sources": [],
            }

        # Embed metadata directly into the context lines so Gemini sees the 'source' and 'context'
        context_lines = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("contract_title", src)
            ctx = doc.metadata.get("context", "General")
            context_lines.append(
                f"CONTRACT: {title} (File: {src}) | SECTION: {ctx} | CONTENT: {doc.page_content}"
            )

        context = "\n".join(context_lines)

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
