from src.extraction.intent_classifier import classify_intent
from src.extraction.ner_engine import extract_entities
from src.extraction.srl_engine import extract_srl


def test_classify_intent():
    assert classify_intent("Bên B phải thanh toán trước ngày 5.") == "Obligation"
    assert classify_intent("Bên A không được đơn phương hủy hợp đồng.") == "Prohibition"
    assert classify_intent("Bên C có quyền yêu cầu đền bù.") == "Right"
    assert (
        classify_intent("Hợp đồng sẽ chấm dứt nếu vi phạm.") == "Termination Condition"
    )


def test_extract_entities():
    text = "Bên B thanh toán 10.000.000 VNĐ vào ngày 15/08/2026."
    entities = extract_entities(text)
    labels = [e["label"] for e in entities]
    assert "PARTY" in labels
    assert "MONEY" in labels
    assert "DATE" in labels


def test_extract_srl():
    text = "Bên A thanh toán tiền."
    entities = [{"text": "Bên A", "label": "PARTY", "span": (0, 5)}]
    dependencies = [
        {"token": "Bên", "relation": "nsubj"},
        {"token": "thanh toán", "relation": "root"},
    ]
    srl = extract_srl(text, entities, dependencies=dependencies)
    assert "predicate" in srl
    assert "roles" in srl
    assert "Agent" in srl["roles"]
