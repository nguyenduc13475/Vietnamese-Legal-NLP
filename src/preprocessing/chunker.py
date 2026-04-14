import re

from src.extraction.ner_engine import extract_ultra_entities


def chunk_np(text: str) -> list[tuple[str, str]]:
    """
    Converts ULTRA-NER labels to IOB format for NP Chunking task.
    Any extracted entity that is not 'PREDICATE' is treated as part of a Noun Phrase (NP).
    """
    if not text or not text.strip():
        return []

    # 1. Get all entities from the unified ULTRA-NER model
    entities = extract_ultra_entities(text)

    # 2. Filter valid NP entities (everything except PREDICATE)
    # Note: 'O' labels are implicitly excluded since extract_ultra_entities ignores them
    valid_entities = [e for e in entities if e.get("label") != "PREDICATE"]

    # 3. Tokenize text into words/syllables, tracking character spans.
    # We use regex \S+ to emulate text.split() while retaining exact start/end indices.
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append(
            {
                "word": match.group(),
                "start": match.start(),
                "end": match.end(),
                "tag": "O",  # Default to Outside
            }
        )

    # 4. Align tokens with valid entity spans using character offset overlap
    for ent in valid_entities:
        ent_start, ent_end = ent["span"]
        is_first_token_in_span = True

        for token in tokens:
            # Check if the token's character span overlaps with the entity's character span
            if token["start"] < ent_end and token["end"] > ent_start:
                if is_first_token_in_span:
                    token["tag"] = "B-NP"
                    is_first_token_in_span = False
                else:
                    token["tag"] = "I-NP"

    # 5. Format output to match the required interface: list of tuples (word, tag)
    iob_results = [(t["word"], t["tag"]) for t in tokens]

    return iob_results
