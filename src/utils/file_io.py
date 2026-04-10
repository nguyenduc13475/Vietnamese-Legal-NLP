import json
import os
from typing import Dict, List, Union


def read_text(file_path: str) -> str:
    """Read content from text file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(file_path: str, data: Union[Dict, List]):
    """Save data to a JSON file using UTF-8 encoding."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
