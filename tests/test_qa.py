import os
import shutil

from src.qa.retriever import LegalRetriever


def test_retriever_ingestion_and_search():
    test_db_dir = "data/vector_db_test"

    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)

    retriever = LegalRetriever(persist_directory=test_db_dir)

    clauses = [
        "Bên A có trách nhiệm thanh toán 10.000.000 VNĐ cho Bên B.",
        "Nghiêm cấm người lao động chia sẻ thông tin mật.",
    ]
    metadata = [{"source": "test_1"}, {"source": "test_2"}]
    retriever.add_clauses(clauses, metadata)

    results = retriever.retrieve("Số tiền cần thanh toán là bao nhiêu?", top_k=1)

    assert len(results) > 0
    assert "10.000.000 VNĐ" in results[0].page_content
    assert results[0].metadata["source"] == "test_1"

    shutil.rmtree(test_db_dir)
