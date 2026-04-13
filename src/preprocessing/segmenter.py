import re


def segment_clauses(text: str) -> list[dict]:
    """
    Extracts independent clauses line-by-line.
    Extracts context tags and ALIASES for RAG metadata.
    """
    raw_lines = text.split("\n")
    clauses = []
    current_context = "General"
    current_aliases = "[]"  # Default empty list string

    # Regex to identify basic list items (1., a), I., -)
    bullet_pattern = re.compile(
        r"^([a-zđ][\)\.]|\d+(?:\.\d+)*[\.\):]?|[IVXLCDM]+[\.\)]|[-+])\s+", re.IGNORECASE
    )

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Detect ALIASES line - skip adding as clause but store for metadata
        if line.startswith("[ALIASES]"):
            current_aliases = line.replace("[ALIASES]", "", 1).strip()
            continue

        # Detect and extract the representative Title marker
        is_title = False
        if line.startswith("[TITLE]"):
            is_title = True
            line = line.replace("[TITLE]", "", 1).strip()

        # Extract context if present (e.g., "[Điều 1] Nội dung...")
        ctx_match = re.match(r"^\[(.*?)\]\s*(.*)", line)
        if ctx_match:
            current_context = ctx_match.group(1).strip()
            line = ctx_match.group(2).strip()

        # Clean up bullet points at the start
        clean_line = bullet_pattern.sub("", line).strip()

        # Discard fragments that are too short to be meaningful
        if len(clean_line) <= 5:
            continue

        # Since one line = one clause, we simply append it directly
        clauses.append(
            {
                "text": clean_line,
                "context": current_context,
                "is_title": is_title,
                "aliases": current_aliases,
            }
        )

    return clauses
