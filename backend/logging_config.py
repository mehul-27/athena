"""Central logging setup.

Observability is a first-class requirement here: upload/extract/chunk/embed/
index/retrieve/context-injection/LLM-call must all be visible in the console so
RAG grounding can be *proven* from the logs rather than assumed.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_NOISY = {
    "chromadb": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "fastembed": logging.WARNING,
    "onnxruntime": logging.WARNING,
}


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for name, lvl in _NOISY.items():
        logging.getLogger(name).setLevel(lvl)
