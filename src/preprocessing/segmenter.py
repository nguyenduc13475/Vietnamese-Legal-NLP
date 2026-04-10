import re

from underthesea import sent_tokenize


def segment_clauses(text: str) -> list[str]:
    """
    Splits contract text into independent clauses.
    Since the raw docs are cleaned by an LLM to guarantee one semantic clause per line,
    this function primarily relies on newline splitting.
    Applies minimal fallback tokenization for excessively long or unformatted lines.
    """
    raw_lines = text.split("\n")
    clauses = []

    # Regex to identify basic list items (1., a), I., -) for light cleaning
    bullet_pattern = re.compile(
        r"^([a-zđ][\)\.]|\d+(?:\.\d+)*[\.\):]?|[IVXLCDM]+[\.\)]|[-+])\s+", re.IGNORECASE
    )

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Clean up bullet points at the start of lines
        clean_line = bullet_pattern.sub("", line).strip()

        # Skip lines that are too short (noise/garbage characters)
        if len(clean_line) <= 5:
            continue

        # Use sent_tokenize to safely split potential multiple sentences on the same line.
        # We sweep through the tokenized parts: if a part has only a few words (e.g., < 4 words),
        # it might be a false split (like "Điều 1.", "a.", "Tp.") so we continue sweeping.
        sentences = sent_tokenize(clean_line)
        current_sentence = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if not current_sentence:
                current_sentence = sent
            else:
                current_sentence += " " + sent

            # If from left to the dot we have enough words, separate into a new sentence
            if len(current_sentence.split()) >= 4:
                clauses.append(current_sentence)
                current_sentence = ""

        # Append any remaining text that didn't reach the word threshold but is valid
        if current_sentence:
            if len(current_sentence) > 2:
                clauses.append(current_sentence)

    return clauses
