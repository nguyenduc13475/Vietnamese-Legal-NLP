import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)
