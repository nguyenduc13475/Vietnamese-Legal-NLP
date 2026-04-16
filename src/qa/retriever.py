import ast

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class LegalRetriever:
    def __init__(self, persist_directory: str = "data/vector_db"):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Using a lightweight multilingual model suitable for Vietnamese, explicitly bound to hardware
        self.embeddings = HuggingFaceEmbeddings(
            model_name="keepitreal/vietnamese-sbert", model_kwargs={"device": device}
        )
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embeddings
        )

    def add_clauses(self, clauses: list[str], metadata: list[dict] = None):
        if not clauses:
            return

        if not metadata:
            metadata = [{"source": "unknown"} for _ in clauses]

        # Standardizing metadata for consistency across all pipeline entry points
        clean_metadata = []
        for meta in metadata:
            # Cast all values to string for ChromaDB compatibility
            clean_metadata.append({k: str(v) for k, v in meta.items()})

        print(f"Indexing {len(clauses)} clauses into {self.persist_directory}...")
        self.vector_store.add_texts(texts=clauses, metadatas=clean_metadata)
        print("Indexing completed successfully.")

    def get_available_sources(self) -> list[str]:
        """Fetch unique document sources currently indexed in the Vector DB."""
        try:
            data = self.vector_store.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            sources = set(
                meta.get("source") for meta in metadatas if meta and "source" in meta
            )
            return sorted(list(sources))
        except Exception as e:
            print(f"Error fetching sources from Vector DB: {e}")
            return []

    def get_sources_with_titles(self) -> dict:
        """Fetch mapping of {filename: contract_title} for LLM routing."""
        try:
            data = self.vector_store.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])

            source_map = {}
            for meta in metadatas:
                if meta and "source" in meta:
                    src = meta["source"]
                    # Map the filename to its title (fallback to filename if title is missing)
                    if src not in source_map:
                        source_map[src] = meta.get("contract_title", src)
            return source_map
        except Exception as e:
            print(f"Error fetching source mappings: {e}")
            return {}

    def _match_with_aliases(
        self, query_val: str, doc_val: str, aliases_str: str
    ) -> float:
        """
        Match two strings considering the alias groups provided in aliases_str.
        Returns a similarity score [0, 1].
        """
        q_clean = query_val.lower().strip()
        d_clean = doc_val.lower().strip()

        if q_clean == d_clean:
            return 1.0

        try:
            if not aliases_str or aliases_str == "[]":
                alias_groups = []
            else:
                alias_groups = ast.literal_eval(aliases_str)
        except Exception:
            alias_groups = []

        # Find if doc_val belongs to any alias group
        target_group = []
        for group in alias_groups:
            if any(str(member).lower().strip() == d_clean for member in group):
                target_group = [str(m).lower().strip() for m in group]
                break

        # If no group found, just do standard comparison
        if not target_group:
            if q_clean in d_clean or d_clean in q_clean:
                return 0.6

            # Word overlap fallback for non-aliases
            q_words = set(q_clean.split())
            d_words = set(d_clean.split())
            overlap = len(q_words & d_words)
            if overlap > 0:
                return (overlap / max(len(q_words), len(d_words))) * 0.4
            return 0.0

        # If group found, compare query_val against EVERY member in the group
        max_score = 0.0
        for alias in target_group:
            if q_clean == alias:
                max_score = 1.0
                break
            if q_clean in alias or alias in q_clean:
                max_score = max(max_score, 0.8)  # Higher score for alias match

            # Simple word overlap for alias members
            q_words = set(q_clean.split())
            a_words = set(alias.split())
            overlap = len(q_words & a_words)
            if overlap > 0:
                max_score = max(
                    max_score, (overlap / max(len(q_words), len(a_words))) * 0.7
                )

        return max_score

    def _calculate_srl_score(
        self, blueprint_srl: dict, doc_metadata: dict, bp_pred_emb: list = None
    ) -> tuple:
        """
        Heuristic rule-based scoring for SRL. Range [0, 1].
        Returns (total_score, breakdown_dict)
        """
        default_breakdown = {
            "predicate_match": 0.0,
            "role_matches": {},
            "role_final_score": 0.0,
        }

        if not blueprint_srl or not isinstance(blueprint_srl, dict):
            return 0.0, default_breakdown

        try:
            doc_pred = str(doc_metadata.get("predicate", "")).lower().strip()
            doc_roles = ast.literal_eval(doc_metadata.get("srl_roles", "{}"))
            aliases_str = str(doc_metadata.get("aliases", "[]"))
        except Exception:
            # Metadata might be corrupted or missing
            return 0.0, default_breakdown

        bp_pred = str(blueprint_srl.get("predicate", "N/A")).lower().strip()
        bp_roles = blueprint_srl.get("roles", {})

        score = 0.0
        # 1. Predicate Similarity (Weight 0.4)
        if bp_pred != "n/a" and doc_pred:
            if bp_pred == doc_pred:
                score += 0.4
            elif bp_pred in doc_pred or doc_pred in bp_pred:
                score += 0.25
            else:
                # Use semantic embedding for synonym matching (e.g., "thanh toán" vs "trả tiền")
                try:
                    # Use pre-calculated query embedding if provided to save API/compute time
                    emb_bp = (
                        bp_pred_emb
                        if bp_pred_emb
                        else self.embeddings.embed_query(bp_pred)
                    )
                    emb_doc = self.embeddings.embed_query(doc_pred)

                    dot_product = sum(a * b for a, b in zip(emb_bp, emb_doc))
                    norm_bp = sum(a * a for a in emb_bp) ** 0.5
                    norm_doc = sum(b * b for b in emb_doc) ** 0.5

                    cos_sim = (
                        (dot_product / (norm_bp * norm_doc))
                        if (norm_bp * norm_doc) > 0
                        else 0.0
                    )

                    if cos_sim > 0.6:
                        score += ((cos_sim - 0.6) / 0.4) * 0.2
                except Exception:
                    # Fallback to Jaccard similarity
                    bp_words, doc_words = set(bp_pred.split()), set(doc_pred.split())
                    overlap = len(bp_words & doc_words)
                    if overlap > 0:
                        score += (overlap / max(len(bp_words), len(doc_words))) * 0.2

        # 2. Roles Similarity (Weight 0.6)
        role_score = 0.0
        bp_active_roles = {
            k: str(v).lower().strip()
            for k, v in bp_roles.items()
            if v and str(v).lower() != "n/a"
        }

        if not bp_active_roles:
            return score  # Return predicate score if no roles requested

        for k, bp_val in bp_active_roles.items():
            doc_val = str(doc_roles.get(k, "")).lower().strip()
            if not doc_val:
                continue

            # Delegate role matching to the alias helper
            match_score = self._match_with_aliases(bp_val, doc_val, aliases_str)
            role_score += match_score

        role_final = (
            (role_score / len(bp_active_roles)) * 0.6 if bp_active_roles else 0.0
        )
        total_score = min(score + role_final, 1.0)

        breakdown = {
            "predicate_match": round(score, 3),
            "role_matches": {
                k: round(match_score, 3)
                for k, match_score in zip(
                    bp_active_roles.keys(),
                    [
                        self._match_with_aliases(
                            bp_val,
                            str(doc_roles.get(rk, "")).lower().strip(),
                            aliases_str,
                        )
                        for rk, bp_val in bp_active_roles.items()
                    ],
                )
            },
            "role_final_score": round(role_final, 3),
        }
        return float(total_score), breakdown

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        source_filters: list[str] = None,
        intent_filters: list[str] = None,
        entity_filters: list[str] = None,
        srl_filter: dict = None,
    ) -> list:
        """
        Retrieval: Hard Filter (Metadata) -> Soft Re-rank (SRL + Vector Norm).
        """
        # Phase 1: Hard Filter (Source & Intent)
        # Fetching a larger candidate pool for re-ranking
        fetch_k = 100
        filters = []

        if source_filters:
            valid_src = [
                s
                for s in source_filters
                if s not in ["All Contracts", "Tất cả Hợp đồng"]
            ]
            if valid_src:
                filters.append(
                    {"source": {"$in": valid_src}}
                    if len(valid_src) > 1
                    else {"source": valid_src[0]}
                )

        if intent_filters:
            valid_intents = [
                i
                for i in intent_filters
                if i
                in [
                    "Obligation",
                    "Prohibition",
                    "Right",
                    "Termination Condition",
                    "Other",
                ]
            ]
            if valid_intents:
                filters.append(
                    {"intent": {"$in": valid_intents}}
                    if len(valid_intents) > 1
                    else {"intent": valid_intents[0]}
                )

        where_filter = (
            {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)
        )

        # Use similarity_search_with_score to get raw distances (Chroma returns L2/IP distances)
        candidates = self.vector_store.similarity_search_with_score(
            query, k=fetch_k, filter=where_filter
        )

        if not candidates:
            return []

        # Normalization: Invert distance to score. Higher is better.
        # Handle the case where distances might be negative or very large.
        raw_distances = [c[1] for c in candidates]
        min_dist, max_dist = min(raw_distances), max(raw_distances)

        normalized_candidates = []
        for doc, dist in candidates:
            # Simple Min-Max normalization for distance -> similarity score [0, 1]
            # If distance is small, score is close to 1.
            norm_vec_score = (
                1.0
                if max_dist == min_dist
                else (max_dist - dist) / (max_dist - min_dist)
            )
            normalized_candidates.append((doc, norm_vec_score))

        # Phase 1.5: Entity Filtering (Hard Filter)
        final_candidates = []
        if entity_filters:
            for doc, v_score in normalized_candidates:
                try:
                    doc_ents_str = doc.metadata.get("entities", "[]")
                    doc_ents = ast.literal_eval(doc_ents_str)
                    doc_labels = [e["label"] for e in doc_ents if isinstance(e, dict)]
                    if any(ef in doc_labels for ef in entity_filters):
                        final_candidates.append((doc, v_score))
                except Exception as e:
                    print(e)
                    continue
        else:
            final_candidates = normalized_candidates

        srl_lambda = 0.7
        re_ranked = []

        # Pre-embed the blueprint predicate once to optimize re-ranking
        bp_pred_emb = None
        if (
            srl_filter
            and srl_filter.get("predicate")
            and srl_filter["predicate"] != "N/A"
        ):
            try:
                bp_pred_emb = self.embeddings.embed_query(srl_filter["predicate"])
            except Exception:
                bp_pred_emb = None

        for doc, v_score in final_candidates:
            srl_score, srl_breakdown = self._calculate_srl_score(
                srl_filter, doc.metadata, bp_pred_emb=bp_pred_emb
            )
            combined_score = v_score + (srl_lambda * srl_score)

            doc.metadata["score_vector"] = round(float(v_score), 4)
            doc.metadata["score_srl"] = round(float(srl_score), 4)
            doc.metadata["score_srl_breakdown"] = str(srl_breakdown)
            doc.metadata["score_total"] = round(float(combined_score), 4)

            re_ranked.append((doc, combined_score))

        re_ranked.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in re_ranked[:top_k]]

    def get_total_count(self) -> int:
        """Get total number of documents in the vector store."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def get_relevant_sources_by_threshold(
        self, query: str, threshold: float = 0.35, max_docs: int = 5
    ) -> list[str]:
        """
        Phase 1 Retrieval: Route query to specific documents by comparing query against [TITLE] clauses.
        Uses a relevance threshold instead of fixed Top-K to allow for dynamic multi-doc matching.
        """
        try:
            # fetch_k is higher to ensure we find potential title matches across the whole collection
            results_with_scores = (
                self.vector_store.similarity_search_with_relevance_scores(
                    query, k=max_docs, filter={"is_title": "True"}
                )
            )

            # results_with_scores is a list of tuples: (Document, Score)
            # Scores are typically normalized [0, 1] in LangChain relevance implementations
            sources = []
            for doc, score in results_with_scores:
                print(
                    f"DEBUG: Found potential source '{doc.metadata.get('source')}' with score {score:.4f}"
                )
                if score >= threshold:
                    sources.append(doc.metadata.get("source"))

            return list(set(sources))  # Return unique sources
        except Exception as e:
            print(f"Phase 1 Routing Error (Threshold mode): {e}")
            # Fallback to standard similarity search if relevance scoring fails/not supported by the specific model config
            results = self.vector_store.similarity_search(
                query, k=max_docs, filter={"is_title": "True"}
            )
            return list(set([doc.metadata.get("source") for doc in results]))

    def get_all_records(self, limit: int = None, offset: int = 0) -> list[dict]:
        """Fetch raw records from the Vector DB with optional pagination."""
        try:
            kwargs = {"include": ["metadatas", "documents"]}
            if limit is not None:
                kwargs["limit"] = limit
            if offset > 0:
                kwargs["offset"] = offset

            # Fetch data directly from the underlying Chroma collection
            data = self.vector_store.get(**kwargs)
            records = []

            if data and "documents" in data:
                for i in range(len(data["documents"])):
                    meta = data["metadatas"][i] if data["metadatas"] else {}
                    records.append(
                        {
                            "id": data["ids"][i] if "ids" in data else str(i),
                            "document": data["documents"][i],
                            "source": meta.get("source", "Unknown"),
                            "contract_title": meta.get("contract_title", "N/A"),
                            "intent": meta.get("intent", "Unknown"),
                            "np_chunks": meta.get("np_chunks", "[]"),
                            "entities": meta.get("entities", "[]"),
                            "predicate": meta.get("predicate", ""),
                            "srl_roles": meta.get("srl_roles", "{}"),
                            "dependencies": meta.get("dependencies", "[]"),
                        }
                    )
            return records
        except Exception as e:
            print(f"Error fetching records from Vector DB: {e}")
            return []

    def delete_document(self, filename: str):
        """Delete all vectors associated with a specific filename."""
        try:
            self.vector_store.delete(where={"source": filename})
            return True
        except Exception as e:
            print(f"Error deleting from Vector DB: {e}")
            return False

    def update_source_name(self, old_source: str, new_source: str):
        """
        Update the 'source' metadata field for all vectors belonging to a file.
        This ensures DB consistency when a processed file is renamed.
        """
        try:
            # 1. Try exact match first
            data = self.vector_store.get(
                where={"source": old_source}, include=["metadatas"]
            )
            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])

            # 2. Fallback: if renaming a .txt but DB has .pdf/.docx (due to ingestion mismatch)
            if not ids:
                base_name = old_source.rsplit(".", 1)[0]
                all_data = self.vector_store.get(include=["metadatas"])
                for i, meta in enumerate(all_data.get("metadatas", [])):
                    db_source = meta.get("source", "")
                    if db_source.startswith(base_name):
                        ids.append(all_data["ids"][i])
                        metadatas.append(meta)

            if not ids:
                print(f"Warning: No vectors found matching source '{old_source}'")
                return True

            # 3. Update metadata locally
            for meta in metadatas:
                meta["source"] = new_source

            # 4. Push updates directly to the underlying Chroma collection.
            # Bypassing Langchain's wrapper which breaks on raw dicts.
            self.vector_store._collection.update(ids=ids, metadatas=metadatas)
            print(
                f"Successfully updated {len(ids)} vectors from '{old_source}' to '{new_source}'"
            )
            return True
        except Exception as e:
            print(f"Error updating source name in Vector DB: {e}")
            return False
