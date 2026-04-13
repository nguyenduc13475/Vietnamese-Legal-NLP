from underthesea import chunk


def chunk_np(text: str) -> list[tuple[str, str]]:
    """
    Extracts Noun Phrases (NP) and labels them using IOB format via underthesea's native chunker.
    Splits compound Vietnamese words into individual syllables for strict IOB representation.
    """
    try:
        chunks = chunk(text)
    except Exception:
        return []

    # Define parts of speech that are typically part of a legal Noun Phrase
    LEGAL_NP_POS = [
        "N",
        "A",
        "M",
        "L",
        "P",
        "Np",
        "Nu",
        "V",
    ]  # Include V for verbal nouns like 'CUNG CẤP'

    refined_chunks = []
    for word, pos, chunk_tag in chunks:
        clean_word = word.replace("_", " ")
        # Rule 1: Strictly filter out non-NP tags unless they are potential legal NP components
        # If underthesea labels it B-VP but it's part of a Title/Entity, we might want it.
        # But for Assignment 1.2, we must prioritize NP tags.
        final_tag = chunk_tag if "NP" in chunk_tag else "O"

        # Rule 2: Force merge common legal verbs/adjectives into the current NP
        if refined_chunks:
            prev = refined_chunks[-1]
            if prev["tag"] in ["B-NP", "I-NP"]:
                # If current word is a legal component, treat as continuation (I-NP)
                if pos in LEGAL_NP_POS:
                    final_tag = "I-NP"

        refined_chunks.append({"word": clean_word, "pos": pos, "tag": final_tag})

    # Convert word-based chunks to syllable-based BIO tags
    syllable_chunks = []
    for item in refined_chunks:
        sub_words = item["word"].split()
        if not sub_words:
            continue

        tag = item["tag"]
        syllable_chunks.append((sub_words[0], tag))

        # Internal syllables of the same word/phrase inherit the continuation tag
        suffix_tag = "I-NP" if tag != "O" else "O"
        for sub_word in sub_words[1:]:
            syllable_chunks.append((sub_word, suffix_tag))

    return syllable_chunks
