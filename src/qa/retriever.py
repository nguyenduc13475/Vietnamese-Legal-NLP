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

    def retrieve(
        self, query: str, top_k: int = 10, source_filters: list[str] = None
    ) -> list:
        """
        Retrieve relevant clauses using Max Marginal Relevance (MMR).
        Supports filtering by multiple source documents simultaneously from Phase 1.
        """
        search_kwargs = {
            "k": top_k,
            "fetch_k": 30,
        }  # Fetch 30 then select 10 most diverse

        if source_filters:
            # Clean out UI wildcard strings if they accidentally get passed
            valid_filters = [
                s
                for s in source_filters
                if s not in ["All Contracts", "Tất cả Hợp đồng"]
            ]

            if len(valid_filters) == 1:
                search_kwargs["filter"] = {"source": valid_filters[0]}
            elif len(valid_filters) > 1:
                # Use ChromaDB's $in operator to search across multiple specific files
                search_kwargs["filter"] = {"source": {"$in": valid_filters}}

        return self.vector_store.max_marginal_relevance_search(query, **search_kwargs)

    def get_total_count(self) -> int:
        """Get total number of documents in the vector store."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def get_top_sources_by_title(self, query: str, top_k: int = 2) -> list[str]:
        """
        Phase 1 Retrieval: Search ONLY against clauses marked as 'is_title'.
        This returns the filenames of the most relevant contracts to the query.
        """
        try:
            results = self.vector_store.similarity_search(
                query, k=top_k, filter={"is_title": "True"}
            )
            # Extract unique sources from the matching title clauses
            sources = set(
                doc.metadata.get("source")
                for doc in results
                if "source" in doc.metadata
            )
            return list(sources)
        except Exception as e:
            print(f"Phase 1 Title Search Error: {e}")
            return []

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
