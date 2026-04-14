from src.extraction.ner_engine import extract_ultra_entities


def chunk_np(text: str) -> list[tuple[str, str]]:
    """
    [Task 1.2] NP Chunking via ULTRA-NER.
    Converts (PARTY, MONEY, DATE, RATE, PENALTY, LAW, OBJECT) to NP boundaries.
    PREDICATE is treated as 'O'.
    """
    if not text:
        return []

    entities = extract_ultra_entities(text)
    # Map all NP-like labels to a generic NP tag for Professor's IOB requirement
    # Using simple split for token display in Task 1.2
    raw_tokens = text.split()

    # We use a character-based matching to ensure sub-word consistency from ULTRA-NER
    results = []
    current_pos = 0
    for word in raw_tokens:
        start = text.find(word, current_pos)
        end = start + len(word)
        current_pos = end

        assigned_tag = "O"
        for ent in entities:
            if ent["label"] == "PREDICATE":
                continue  # Predicate is not NP

            e_start, e_end = ent["span"]
            if start >= e_start and end <= e_end:
                assigned_tag = "B-NP" if start == e_start else "I-NP"
                break
        results.append((word, assigned_tag))

    return results
