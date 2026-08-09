"""Pure loaders for the language-specific reference scripts."""

import json
from pathlib import Path


SCRIPT_DIRECTORY = Path("script")


def load_closed_script(language: str = "en") -> list[dict]:
    path = SCRIPT_DIRECTORY / f"{language}.json"
    if not path.exists():
        path = SCRIPT_DIRECTORY / "en.json"
    with path.open(encoding="utf-8") as script_file:
        return json.load(script_file)
