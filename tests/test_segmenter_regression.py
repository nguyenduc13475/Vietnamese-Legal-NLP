from src.preprocessing.segmenter import segment_clauses


def test_llm_formatted_text():
    """Ensure that text pre-formatted by Gemini (one clause per line) is parsed cleanly."""
    text = "Điều 1. Phạm vi hợp đồng\nBên A bán cho Bên B một căn nhà.\nGiá trị căn nhà là 10 tỷ đồng."
    clauses = segment_clauses(text)
    assert len(clauses) == 3
    assert "Điều 1. Phạm vi hợp đồng" in clauses[0]
    assert "Giá trị căn nhà là 10 tỷ đồng" in clauses[2]


def test_raw_paragraph_fallback():
    """Ensure that a raw, unformatted paragraph is still split via the sent_tokenize fallback."""
    text = (
        "Bên A bán căn nhà cho Bên B. Bên B đồng ý mua căn nhà đó! Hai bên cùng ký tên."
    )
    clauses = segment_clauses(text)
    assert len(clauses) == 3
    assert "Bên A bán căn nhà" in clauses[0]
    assert "Hai bên cùng ký tên" in clauses[2]


def test_bullet_cleaning():
    """Ensure basic bullet points are stripped by the minimal regex."""
    text = "a) Người thuê lại trả chậm hơn 1 tháng.\n1.1. Vi phạm trật tự."
    clauses = segment_clauses(text)
    assert len(clauses) == 2
    assert clauses[0] == "Người thuê lại trả chậm hơn 1 tháng."
    assert clauses[1] == "Vi phạm trật tự."
