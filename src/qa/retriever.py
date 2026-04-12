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
        if not metadata:
            metadata = [{"source": "unknown"} for _ in clauses]

        # Ensure all values in metadata are strings to avoid ChromaDB Schema errors
        clean_metadata = []
        for meta in metadata:
            clean_metadata.append({k: str(v) for k, v in meta.items()})

        self.vector_store.add_texts(texts=clauses, metadatas=clean_metadata)
        # self.vector_store.persist() is no longer needed/supported in langchain-chroma.
        # It persists automatically when persist_directory is provided during initialization.

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

    def retrieve(self, query: str, top_k: int = 7, source_filter: str = None) -> list:
        """Retrieve relevant clauses, optionally filtered by a specific source document."""
        if source_filter and source_filter != "All Contracts":
            return self.vector_store.similarity_search(
                query, k=top_k, filter={"source": source_filter}
            )
        return self.vector_store.similarity_search(query, k=top_k)

    def get_total_count(self) -> int:
        """Get total number of documents in the vector store."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

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
