import logging

import stanza
import torch

USE_GPU = torch.cuda.is_available()
_nlp_pipeline = None


def get_pipeline():
    """
    Lazy initialization of Stanza pipeline.
    It will automatically download the 'vi' model if it doesn't exist.
    """
    global _nlp_pipeline
    if _nlp_pipeline is None:
        try:
            _nlp_pipeline = stanza.Pipeline(
                "vi",
                processors="tokenize,pos,lemma,depparse",
                use_gpu=USE_GPU,
                verbose=False,
                logging_level="ERROR",
            )
        except Exception:
            stanza.download("vi", verbose=False, logging_level="ERROR")
            _nlp_pipeline = stanza.Pipeline(
                "vi",
                processors="tokenize,pos,lemma,depparse",
                use_gpu=USE_GPU,
                verbose=False,
                logging_level="ERROR",
            )
    return _nlp_pipeline


def parse_dependency(text: str) -> list[dict]:
    """
    Utilize Stanza for practical Vietnamese dependency parsing.
    """
    # Handle exception if the clause is empty
    if not text.strip():
        return []

    result = []
    try:
        nlp = get_pipeline()
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
