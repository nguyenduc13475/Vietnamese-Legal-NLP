import logging
import ssl

import stanza
import torch

# Bypass SSL verification if behind a strict firewall/proxy
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

USE_GPU = torch.cuda.is_available()
_nlp_pipeline = None


def get_pipeline():
    """
    Lazy initialization of Stanza pipeline.
    Forces offline mode first to prevent SSL/Network crashes.
    """
    global _nlp_pipeline
    if _nlp_pipeline is None:
        try:
            # TRY OFFLINE FIRST: REUSE_RESOURCES prevents it from hitting GitHub
            _nlp_pipeline = stanza.Pipeline(
                "vi",
                processors="tokenize,pos,lemma,depparse",
                use_gpu=USE_GPU,
                verbose=False,
                logging_level="ERROR",
                download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            )
        except Exception:
            # If offline fails (model not downloaded yet), force download
            print("Downloading Stanza 'vi' model (This only happens once)...")
            stanza.download("vi", verbose=False, logging_level="ERROR")
            _nlp_pipeline = stanza.Pipeline(
                "vi",
                processors="tokenize,pos,lemma,depparse",
                use_gpu=USE_GPU,
                verbose=False,
                logging_level="ERROR",
                download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            )
    return _nlp_pipeline


def parse_dependency(text: str) -> list[dict]:
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
