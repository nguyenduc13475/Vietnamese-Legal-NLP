import io
import re
import unicodedata

import docx
from pypdf import PdfReader


def extract_text_from_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="ignore")


def extract_text_from_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"[Error] Lỗi khi đọc file PDF: {str(e)}"


def extract_text_from_docx(content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"[Error] Error reading DOCX file: {str(e)}"


def clean_contract_text(text: str) -> str:
    """
    Advanced text cleaning specifically tuned for Vietnamese Legal Contracts.
    """
    # 1. Unicode Normalization (NFC) - Crucial for Vietnamese to
    # prevent tokenization errors in PhoBERT/Underthesea
    text = unicodedata.normalize("NFC", text)

    # 2. Remove hidden characters (zero-width spaces, soft hyphens)
    # generated during the PDF-to-Text conversion process
    text = re.sub(r"[\u200B-\u200D\uFEFF\u00AD]", "", text)

    # 3. Remove Table of Contents (TOC) - Lines with consecutive dots ending in a number
    text = re.sub(r"(?m)^.*\.{4,}\s*\d+\s*$", "", text)

    # 4. Remove page numbering formats and horizontal separators
    text = re.sub(
        r"(?i)^\s*(?:page|trang)\s*\d+\s*(?:of|/)?\s*\d*\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*[-_=]{3,}\s*$", "", text)  # Remove horizontal separators

    # 5. Whitespace Normalization: Collapse multiple spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    lines = text.split("\n")
    cleaned_lines = []

    # Regex to identify legal lists/sections to PREVENT improper merging
    bullet_pattern = re.compile(
        r"^([a-zđ][\)\.]|\d+(?:\.\d+)*[\.\):]?|[IVXLCDM]+[\.\)]|[-+]|Điều\s+\d+|Khoản\s+\d+|Mục\s+\d+|Phần\s+\d+)",
        re.IGNORECASE,
    )

    for line in lines:
        line = line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        if cleaned_lines and cleaned_lines[-1] != "":
            prev_line = cleaned_lines[-1]

            # Smart Sentence Wrapping logic
            ends_with_punctuation = re.search(r"[.:;!?]$", prev_line)
            is_all_caps = prev_line.isupper()
            starts_with_bullet = bullet_pattern.match(line)

            # Indicators of broken sentences: Next line begins with a lowercase letter or a comma
            starts_with_lower_or_comma = re.match(
                r"^[,\s]*[a-zđáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]",
                line,
            )

            # Indicators of a broken previous line: Ends with a preposition or conjunction
            ends_with_conjunction = re.search(
                r"(?i)(?:và|hoặc|của|cho|bởi|như|là|thì|mà|để|với|tại|từ|về|thuộc)\s*$",
                prev_line,
            )

            if not ends_with_punctuation and not is_all_caps and not starts_with_bullet:
                if starts_with_lower_or_comma or ends_with_conjunction:
                    # Append current line to the previous line
                    cleaned_lines[-1] = prev_line + " " + line
                    continue

        cleaned_lines.append(line)

    final_text = "\n".join(cleaned_lines)

    # 6. Consolidate blank lines (Reduce 3+ consecutive blank
    # lines to a maximum of 2 to preserve paragraph formatting)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)

    return final_text.strip()


def parse_and_clean_document(file_content: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        text = extract_text_from_pdf(file_content)
    elif ext in ["doc", "docx"]:
        text = extract_text_from_docx(file_content)
    else:
        text = extract_text_from_txt(file_content)

    return clean_contract_text(text)
