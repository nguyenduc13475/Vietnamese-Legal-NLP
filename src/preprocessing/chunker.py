from underthesea import chunk


def chunk_np(text: str) -> list[tuple[str, str]]:
    """
    Extracts Noun Phrases (NP) and labels them using IOB format via underthesea's native chunker.
    """
    try:
        chunks = chunk(text)
    except Exception:
        return []

    chunked = []
    for word, _, chunk_tag in chunks:
        word_clean = word.replace("_", " ")
        # We only care about Noun Phrases (NP)
        if chunk_tag in ["B-NP", "I-NP"]:
            chunked.append((word_clean, chunk_tag))
        else:
            chunked.append((word_clean, "O"))

    return chunked
