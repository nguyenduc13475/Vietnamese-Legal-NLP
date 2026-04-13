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
    temp_chunks = []
    for word, pos, chunk_tag in chunks:
        clean_word = word.replace("_", " ")
        temp_chunks.append({"word": clean_word, "pos": pos, "tag": chunk_tag})

    # Join adjacent NPs if they form a logical unit (Noun + Adj/Noun)
    # e.g., "năng lực" (B-NP) + "thực tế" (B-NP) -> one single NP unit
    refined_chunks = []
    for i in range(len(temp_chunks)):
        curr = temp_chunks[i]
        if i > 0 and curr["tag"] == "B-NP":
            prev = temp_chunks[i - 1]
            # If the current B-NP is a Noun or Adj following another NP, convert to I-NP
            if prev["tag"] in ["B-NP", "I-NP"] and curr["pos"] in ["N", "A", "M", "L"]:
                curr["tag"] = "I-NP"
        refined_chunks.append(curr)

    chunked = []
    for item in refined_chunks:
        sub_words = item["word"].split()
        if not sub_words:
            continue

        # Maintain BIO continuity within the word itself
        tag = item["tag"]
        chunked.append((sub_words[0], tag))
        # Subsequent syllables of the same word are always Inside (I-NP) if the word started a chunk
        suffix_tag = "I-NP" if tag != "O" else "O"
        for sub_word in sub_words[1:]:
            chunked.append((sub_word, suffix_tag))

    return chunked
