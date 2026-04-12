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
        target_sources = None
        intent_filters = None
        entity_filters = None
        routing_debug_info = (
            "Phase 1 skipped: Manual source filter applied or no routing required."
        )
        debug_prompt = "Phase 2 Prompt was not generated due to missing context."

        # Phase 1: LLM-Based Query Routing & Search Blueprint Generation
        if source_filter and source_filter not in ["All Contracts", "Tất cả Hợp đồng"]:
            target_sources = [source_filter]
        else:
            source_mapping = self.retriever.get_sources_with_titles()
            available_sources = list(source_mapping.keys())

            if available_sources and self.client:
                catalog = "\n".join(
                    [
                        f"- {src} (Title: {title})"
                        for src, title in source_mapping.items()
                    ]
                )
                routing_prompt = (
                    "You are a high-precision Legal Search Architect.\n"
                    f"User Question: '{question}'\n\n"
                    f"Available Documents:\n{catalog}\n\n"
                    "Your task is to generate a 'Search Blueprint' in JSON format to filter relevant clauses from the database.\n"
                    "JSON Fields:\n"
                    "- sources: List of filenames (e.g., ['a.txt']) or [] if all documents apply.\n"
                    "- intents: List of intents likely containing the answer. Choose from: ['Obligation', 'Prohibition', 'Right', 'Termination Condition', 'Other'].\n"
                    "- entity_types: List of entity types mentioned. Choose from: ['PARTY', 'MONEY', 'DATE', 'RATE', 'PENALTY', 'LAW'].\n"
                    "- srl: An object describing the expected semantic roles: {'predicate': 'verb', 'roles': {'Agent': 'who', 'Recipient': 'to whom', 'Theme': 'what', 'Time': 'when', 'Penalty_Rate': 'how much'}}. Use 'N/A' if unknown.\n"
                    "- search_query: A condensed version of the question for vector search. Include entities or specific legal predicates.\n\n"
                    "Return ONLY the JSON object. No explanation."
                )

                try:
                    import json

                    route_response = self.client.models.generate_content(
                        model="gemini-3.1-flash-lite-preview",
                        contents=routing_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0, response_mime_type="application/json"
                        ),
                    )
                    blueprint = json.loads(route_response.text)
                    routing_debug_info = f"--- ROUTING PROMPT ---\n{routing_prompt}\n\n--- SEARCH BLUEPRINT ---\n{json.dumps(blueprint, indent=2, ensure_ascii=False)}"

                    target_sources = (
                        blueprint.get("sources") if blueprint.get("sources") else None
                    )
                    intent_filters = blueprint.get("intents")
                    entity_filters = blueprint.get("entity_types")
                    srl_filter = blueprint.get("srl")
                    question = blueprint.get("search_query", question)
                except Exception as e:
                    routing_debug_info = f"Routing Error: {str(e)}"
            else:
                routing_debug_info = "Phase 1 skipped: No available sources in database or API client missing."

        # Phase 2: Targeted Retrieval
        docs = self.retriever.retrieve(
            query=question,
            top_k=10,
            source_filters=target_sources,
            intent_filters=intent_filters,
            entity_filters=entity_filters,
            srl_filter=srl_filter,
        )

        if not docs:
            return {
                "question": question,
                "answer": "Sorry, I could not find any information in the contract related to your query.",
                "sources": [],
                "routing_debug": routing_debug_info,
                "debug_prompt": "No context retrieved. Generation aborted.",
            }

        # Phase 3: Contextual Generation
        context_lines = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("contract_title", src)
            ctx = doc.metadata.get("context", "General")
            context_lines.append(
                f"SOURCE: {title} ({src}) | SECTION: {ctx} | CONTENT: {doc.page_content}"
            )

        context = "\n".join(context_lines)
        debug_prompt = self.prompt_template.format(context=context, question=question)

        if not self.client:
            answer_text = "[Error: API Key not set]"
        else:
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    contents=debug_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=2048
                    ),
                )
                answer_text = response.text.strip()
            except Exception as e:
                answer_text = f"LLM Error: {str(e)}"

        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ],
            "debug_prompt": debug_prompt,
            "routing_debug": routing_debug_info,
        }
