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

    def retrieve(self, query: str, top_k: int = 7) -> list:
        # Hardcode filter để ép nó CHỈ tìm trong hợp đồng thuê nhà
        # (Sau này bạn có thể truyền biến filter này từ UI xuống)
        return self.vector_store.similarity_search(
            query, k=top_k, filter={"source": "hop_dong_thue_nha.txt"}
        )
