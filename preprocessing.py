"""
preprocessing.py
----------------
Text-cleaning utilities that mirror the preprocessing applied during training.

Notebook Cell 6 applied these steps before tokenisation::

    text = str(text).lower().strip()
    text = re.sub(r'https?://S+|www.S+', ' ', text)  # strip URLs
    text = re.sub(r'@w+', ' ', text)                 # strip @mentions
    text = re.sub(r's+', ' ', text).strip()           # normalise whitespace

No additional transformations were applied before tokenisation.
"""

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    """
    Clean raw user text to match the preprocessing used during training.

    Steps (in order):
      1. Cast to str and strip leading/trailing whitespace.
      2. Lower-case the text.
      3. Remove URLs (http/https/www).
      4. Remove @mentions.
      5. Collapse multiple whitespace characters into a single space.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text ready for tokenisation.
    """
    text = str(text).lower().strip()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
