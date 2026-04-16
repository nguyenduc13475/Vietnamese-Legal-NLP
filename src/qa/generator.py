import json
import time

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

    def _call_gemini_with_fallback(
        self, contents, temperature=0.0, response_mime_type=None, max_output_tokens=None
    ):
        """
        Calls Gemini API with a multi-model fallback and retry mechanism.
        - Tries a list of models sequentially.
        - Waits 5s between model attempts.
        - Waits 60s if a full roll (all models) fails.
        - Throws exception after 5 full rolls.
        """
        models_to_try = [
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash-lite-preview-09-2025",
            "gemma-3-27b-it",
        ]
        max_full_rolls = 5
        last_error = None

        for roll in range(max_full_rolls):
            for model_name in models_to_try:
                try:
                    config_args = {"temperature": temperature}
                    if response_mime_type:
                        config_args["response_mime_type"] = response_mime_type
                    if max_output_tokens:
                        config_args["max_output_tokens"] = max_output_tokens

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_args),
                    )
                    return response
                except Exception as e:
                    last_error = e
                    print(
                        f"[RAG API Warning] Model {model_name} failed. Retrying in 5s... Error: {str(e)}"
                    )
                    time.sleep(5)

            if roll < max_full_rolls - 1:
                print(
                    f"[RAG API Warning] All models failed in roll {roll + 1}/{max_full_rolls}. Waiting 60s before next roll..."
                )
                time.sleep(60)

        raise Exception(
            f"Gemini API completely failed after {max_full_rolls} full rolls. Last error: {str(last_error)}"
        )

    def ask(self, question: str, source_filter: str = None) -> dict:
        # Initialize filters to None to prevent UnboundLocalError
        target_sources = None
        intent_filters = None
        entity_filters = None
        srl_filter = None
        search_query = question  # Default to original question
        routing_debug_info = (
            "Phase 1 skipped: Manual source filter applied or no routing required."
        )
        debug_prompt = "Phase 2 Prompt was not generated due to missing context."

        # Phase 1: LLM-Based Query Routing & Search Blueprint Generation
        source_mapping = self.retriever.get_sources_with_titles()
        available_sources = list(source_mapping.keys())

        # Determine if we are restricted to one source by user or if LLM should decide
        is_fixed_source = source_filter and source_filter not in [
            "All Contracts",
            "Tất cả Hợp đồng",
        ]

        if self.client and available_sources:
            # Build catalog for LLM to understand document context
            catalog_info = ""
            if is_fixed_source:
                # Tell LLM we are focused on this specific document title
                fixed_title = source_mapping.get(source_filter, source_filter)
                catalog_info = f"Search Scope: SPECIFIC DOCUMENT ONLY -> '{fixed_title}' ({source_filter})"
            else:
                # Provide full catalog for multi-doc picking
                catalog_info = (
                    "Search Scope: MULTIPLE DOCUMENTS. Available:\n"
                    + "\n".join(
                        [
                            f"- {src} (Title: {title})"
                            for src, title in source_mapping.items()
                        ]
                    )
                )

            routing_prompt = (
                "You are a high-precision Legal Search Architect.\n"
                f"User Question: '{question}'\n\n"
                f"{catalog_info}\n\n"
                "Task: Generate a 'Search Blueprint' in JSON to extract relevant data using metadata filters.\n"
                "JSON Fields:\n"
                "- sources: (List) Only populate if searching multiple documents. If scope is specific, return [].\n"
                "- intents: (List) Expected clause types: ['Obligation', 'Prohibition', 'Right', 'Termination Condition', 'Other'].\n"
                "- entity_types: (List) Key entities mentioned: ['PARTY', 'MONEY', 'DATE', 'RATE', 'PENALTY', 'LAW'].\n"
                "- srl: (Object) Semantic roles: {'predicate': 'verb', 'roles': {'AGENT': 'who', 'RECIPIENT': 'to whom', 'THEME': 'what', 'TIME': 'when', 'NAME': 'name or identifier', 'CONDITION': 'conditions for that action to be taken', 'LOCATION': 'related location', 'METHOD': 'by what method was this action performed', 'ABOUT': 'the related subject or purpose of the action', 'TRAIT': 'other important information that gives more detailed description of the action'}}. Use 'N/A' for unknown. Should not be too long-winded.\n"
                "- search_query: (String) Optimized Vietnamese keyword string for vector search.\n\n"
                "Return ONLY the JSON object. No explanation."
            )

            try:
                route_response = self._call_gemini_with_fallback(
                    contents=routing_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
                blueprint = json.loads(route_response.text)
                routing_debug_info = f"--- ROUTING PROMPT ---\n{routing_prompt}\n\n--- SEARCH BLUEPRINT ---\n{json.dumps(blueprint, indent=2, ensure_ascii=False)}"

                # Apply logic: User selection overrides LLM source selection
                if is_fixed_source:
                    target_sources = [source_filter]
                else:
                    target_sources = (
                        blueprint.get("sources") if blueprint.get("sources") else None
                    )

                # Semantic filters are ALWAYS applied from LLM blueprint
                intent_filters = blueprint.get("intents")
                entity_filters = blueprint.get("entity_types")
                srl_filter = blueprint.get("srl")
                # Extract optimized query for vector search, keeping original question intact
                search_query = blueprint.get("search_query", question)

            except Exception as e:
                routing_debug_info = f"Routing Error: {str(e)}"
                if is_fixed_source:
                    target_sources = [source_filter]
        else:
            routing_debug_info = "Phase 1 skipped: Missing API client or empty DB."
            if is_fixed_source:
                target_sources = [source_filter]

        # Phase 2: Targeted Retrieval with safe parameter passing
        try:
            docs = self.retriever.retrieve(
                query=search_query,  # Use the optimized search query here
                top_k=10,
                source_filters=target_sources,
                intent_filters=intent_filters,
                entity_filters=entity_filters,
                srl_filter=srl_filter if isinstance(srl_filter, dict) else {},
            )
        except Exception as e:
            print(f"Retrieval error: {e}")
            # Fallback to simple vector search if metadata filtering fails
            docs = self.retriever.retrieve(query=search_query, top_k=5)

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
                response = self._call_gemini_with_fallback(
                    contents=debug_prompt, temperature=0.0, max_output_tokens=2048
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
