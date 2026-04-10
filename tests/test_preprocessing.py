from src.preprocessing.chunker import chunk_np
from src.preprocessing.parser import parse_dependency
from src.preprocessing.segmenter import segment_clauses


def test_segment_clauses():
    text = "Bên A đồng ý bán và Bên B đồng ý mua. Nếu vi phạm, phạt 5%."
    clauses = segment_clauses(text)
    assert len(clauses) >= 2
    assert "phạt 5%" in clauses[1]


def test_chunk_np():
    text = "Bên B thanh toán tiền"
    chunks = chunk_np(text)
    assert len(chunks) > 0
    tags = [tag for _, tag in chunks]
    assert all(tag in ["B-NP", "I-NP", "O"] for tag in tags)


def test_parse_dependency():
    text = "Bên A thanh toán tiền."
    deps = parse_dependency(text)
    assert isinstance(deps, list)
    if len(deps) > 0:
        assert "token" in deps[0]
        assert "head_index" in deps[0]
        assert "head_token" in deps[0]
        assert "relation" in deps[0]
