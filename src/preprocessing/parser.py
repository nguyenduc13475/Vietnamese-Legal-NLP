import logging

import stanza
import torch

# Initialize Stanza pipeline globally so it doesn't reload on every API call.
# It will automatically download the 'vi' model if it doesn't exist.


USE_GPU = torch.cuda.is_available()

try:
    nlp = stanza.Pipeline(
        "vi",
        processors="tokenize,pos,lemma,depparse",
        use_gpu=USE_GPU,
        verbose=False,
        logging_level="ERROR",
    )
except Exception:
    stanza.download("vi", verbose=False, logging_level="ERROR")
    nlp = stanza.Pipeline(
        "vi",
        processors="tokenize,pos,lemma,depparse",
        use_gpu=USE_GPU,
        verbose=False,
        logging_level="ERROR",
    )


def parse_dependency(text: str) -> list[dict]:
    """
    Utilize Stanza for practical Vietnamese dependency parsing.
    """
    # Handle exception if the clause is empty
    if not text.strip():
        return []

    result = []
    try:
        doc = nlp(text)
        for sentence in doc.sentences:
            for word in sentence.words:
                head_token = (
                    sentence.words[word.head - 1].text if word.head > 0 else "ROOT"
                )
                result.append(
                    {
                        "id": word.id,
                        "token": word.text.replace("_", " "),
                        "head_index": word.head,
                        "head_token": head_token.replace("_", " "),
                        "relation": word.deprel,
                    }
                )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(
            f"Stanza Parser error while analyzing sentence '{text[:30]}...': {e}"
        )

    return result
