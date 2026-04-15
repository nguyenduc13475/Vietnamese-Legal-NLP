import re

from src.extraction.ner_engine import extract_ultra_entities


def chunk_np(text: str) -> list[tuple[str, str]]:
    """
    [Task 1.2] NP Chunking.
    Consistent with PhoBERT spans. Merges adjacent NP-type entities into single sequences.
    """
    if not text:
        return []

    # 1. Get all entities from the model
    entities = extract_ultra_entities(text)

    # 2. Identify NP-type indices (Character-level mask)
    # 1 = B-NP, 2 = I-NP, 0 = O
    char_mask = [0] * len(text)

    # Sort entities by start position to process in order
    entities.sort(key=lambda x: x["span"][0])

    last_end = -1
    for ent in entities:
        if ent["label"] == "PREDICATE":
            continue

        start, end = ent["span"]

        # Check if this NP is adjacent to the previous one (merging logic)
        # If the gap is only whitespace, we treat it as continuous
        gap = text[last_end:start] if last_end != -1 else "N/A"
        is_adjacent = last_end != -1 and (last_end == start or not gap.strip())

        for i in range(start, end):
            if i == start and not is_adjacent:
                char_mask[i] = 1  # B-NP
            else:
                char_mask[i] = 2  # I-NP
        last_end = end

    # 3. Tokenize based on the mask to ensure alignment
    # We use a simple regex to find words/punctuation but keep char indices
    results = []
    # Regex logic:
    # 1. (\S+?)(?=\.?\s|$) -> Match non-whitespace greedily but stop before a potential trailing dot.
    # 2. (\.)(?=\s|$) -> Capture that trailing dot as its own token.
    # 3. (\S+) -> Fallback for anything else (like internal dots in IDs).
    pattern = re.compile(r"(\S+?)(?=\.\s|\.$)|(\.)(?=\s|$)|(\S+)")

    for match in pattern.finditer(text):
        word = match.group()
        start, end = match.span()

        # Determine tag based on the mask of characters within this word
        word_mask = char_mask[start:end]

        if 1 in word_mask:
            tag = "B-NP"
        elif 2 in word_mask:
            # If the word starts mid-span, it's I-NP
            tag = "I-NP"
        else:
            tag = "O"

        results.append((word, tag))

    # 4. Final Pass: Correct "I-NP" that should be "B-NP" due to merging
    # and ensure the sequence B-I-I is consistent.
    final_results = []
    for i, (word, tag) in enumerate(results):
        if tag == "I-NP":
            # If it's the very first token or preceded by an "O", force it to B-NP
            if i == 0 or results[i - 1][1] == "O":
                tag = "B-NP"
        final_results.append((word, tag))

    return final_results
