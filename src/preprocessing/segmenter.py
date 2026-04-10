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

        # Since Gemini formats each clause on a single line, we prioritize preserving line integrity.
        # Heuristic: Only activate Underthesea if the line is excessively long (> 40 words)
        # AND shows signs of containing multiple sentences (sentence-ending punctuation + next letter capitalized).
        word_count = len(clean_line.split())
        contains_multiple_sentences = re.search(
            r"[.!?]\s+[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ]",
            clean_line,
        )

        if word_count > 40 and contains_multiple_sentences:
            sentences = sent_tokenize(clean_line)
            for sent in sentences:
                clean_sent = sent.strip()
                if len(clean_sent) > 5:
                    clauses.append(clean_sent)
        else:
            clauses.append(clean_line)

    return clauses
