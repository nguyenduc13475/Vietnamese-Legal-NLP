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

    chunked = []
    for word, _, chunk_tag in chunks:
        # Normalize: replace underscores with spaces and split into individual syllables
        sub_words = word.replace("_", " ").split()

        if not sub_words:
            continue

        if chunk_tag == "B-NP":
            # The very first syllable of a B-NP block gets B-NP, others get I-NP
            chunked.append((sub_words[0], "B-NP"))
            for sub_word in sub_words[1:]:
                chunked.append((sub_word, "I-NP"))
        elif chunk_tag == "I-NP":
            # All syllables in an I-NP block must be I-NP
            for sub_word in sub_words:
                chunked.append((sub_word, "I-NP"))
        else:
            # Everything else (O) remains O
            for sub_word in sub_words:
                chunked.append((sub_word, "O"))

    return chunked
