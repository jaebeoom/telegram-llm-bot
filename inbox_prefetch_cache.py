"""Compatibility wrapper for the src package layout."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

sys.modules[__name__] = importlib.import_module("telegram_llm_bot.inbox_prefetch_cache")
