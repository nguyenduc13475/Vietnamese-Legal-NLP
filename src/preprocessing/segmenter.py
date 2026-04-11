import re


def segment_clauses(text: str) -> list[str]:
    """
    Splits contract text into independent clauses based on a period
    followed by a space, avoiding splits in numbers or emails.
    """
    raw_lines = text.split("\n")
    clauses = []

    # Regex to identify basic list items (1., a), I., -)
    bullet_pattern = re.compile(
        r"^([a-zđ][\)\.]|\d+(?:\.\d+)*[\.\):]?|[IVXLCDM]+[\.\)]|[-+])\s+", re.IGNORECASE
    )

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Clean up bullet points at the start
        clean_line = bullet_pattern.sub("", line).strip()

        if len(clean_line) <= 5:
            continue

        # REGEX EXPLANATION:
        # (?<=\.) : Lookbehind - find a period
        # \s+     : Followed by one or more spaces
        # This splits "Hello. World" into ["Hello.", "World"]
        # but leaves "10.000.000" alone.
        sentences = re.split(r"(?<=\.)\s+", clean_line)

        current_sentence = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            if not current_sentence:
                current_sentence = sent
            else:
                current_sentence += " " + sent

            # Threshold check: only flush the sentence if it has >= 4 words
            # to avoid fragments like "Tp. HCM" being isolated.
            if len(current_sentence.split()) >= 4:
                clauses.append(current_sentence)
                current_sentence = ""

        # Final cleanup for remaining text
        if current_sentence and len(current_sentence) > 2:
            clauses.append(current_sentence)

    return clauses
