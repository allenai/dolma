"""

Utilities for code-related taggers.

@akshitab, @soldni

"""

import json
import logging
from pathlib import Path
from typing import Dict, Generator

import regex
import smart_open
from bs4 import BeautifulSoup  # pylint: disable=import-error
from detect_secrets.core.potential_secret import PotentialSecret
from detect_secrets.core.scan import _process_line_based_plugins
from detect_secrets.core.secrets_collection import SecretsCollection
from detect_secrets.settings import default_settings, get_plugins

logger = logging.getLogger(__name__)


def scan_code(code: str) -> Generator["PotentialSecret", None, None]:
    if not get_plugins():
        logger.error("No plugins to scan with!")
        return

    has_secret = False
    for lines in [code.splitlines()]:
        for secret in _process_line_based_plugins(
            lines=list(enumerate(lines, start=1)),
            filename="code_str.yml",
        ):
            has_secret = True
            yield secret

        if has_secret:
            break


class SecretsCollectionForStringInput(SecretsCollection):
    def scan_str(self, code_str: str):
        for secret in scan_code(code_str):
            self["code_str.yml"].add(secret)


def get_secrets(code: str):
    secrets = SecretsCollectionForStringInput()
    with default_settings():
        secrets.scan_str(code)

    return secrets


def filter_html(html: str) -> float:
    """Filter HTML files based on displayed text VS code ratio"""
    try:
        soup = BeautifulSoup(html, features="html.parser")
    except (TypeError, UnboundLocalError):
        return False

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()
    ratio = len(text) / len(html)

    return (ratio) * (len(text) > 100)


def get_whitespace_regex() -> regex.Pattern:
    return regex.compile(r"\w+|[^\w\s]+")


def get_ext_to_lang_mapping() -> Dict[str, str]:
    path = Path(__file__).parent / "../../data/ext_to_lang_mapping.json"
    with smart_open.open(path, "r") as f:
        return json.load(f)
