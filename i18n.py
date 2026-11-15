"""
Minimal two-language support (English default, Turkish optional).

Usage
-----
    from i18n import L, pick, LocalizedMapping, get_lang

    st.button(L("Run analysis", "Analizi çalıştır"))          # chosen at call time
    label = L(f"{n} months", f"{n} ay")                        # f-strings are fine

Module-level constants that are built at import time cannot call L() (the language
is not known yet). Store their text as {"en": ..., "tr": ...} pairs and either wrap
the whole dict in LocalizedMapping (resolved on every access) or call pick(value)
where it is displayed.

The language lives in st.session_state["lang"] ("en" or "tr"). Outside a Streamlit
run (tests, scripts) the default comes from the EDD_LANG environment variable, else "en".
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Iterator

LANGS = ("en", "tr")
LANG_LABELS = {"en": "English", "tr": "Türkçe"}


def default_lang() -> str:
    env = os.environ.get("EDD_LANG", "en").strip().lower()
    return env if env in LANGS else "en"


def get_lang() -> str:
    lang = None
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is not None:  # only inside a Streamlit run
            import streamlit as st

            lang = st.session_state.get("lang")
    except Exception:
        lang = None
    return lang if lang in LANGS else default_lang()


def L(en: str, tr: str) -> str:
    """Return the English or Turkish text for the current language."""
    return tr if get_lang() == "tr" else en


def _is_pair(v: Any) -> bool:
    return isinstance(v, dict) and len(v) == 2 and set(v.keys()) == {"en", "tr"}


def pick(v: Any) -> Any:
    """Resolve {"en","tr"} pairs (recursively inside dicts, lists and tuples)."""
    if _is_pair(v):
        return v["tr"] if get_lang() == "tr" else v["en"]
    if isinstance(v, dict):
        return {k: pick(x) for k, x in v.items()}
    if isinstance(v, list):
        return [pick(x) for x in v]
    if isinstance(v, tuple):
        return tuple(pick(x) for x in v)
    return v


class LocalizedMapping(Mapping):
    """Read-only view of a dict whose text values are {"en","tr"} pairs; values are resolved on access."""

    def __init__(self, raw: dict):
        self._raw = raw

    def __getitem__(self, key: Any) -> Any:
        return pick(self._raw[key])

    def __iter__(self) -> Iterator:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def __repr__(self) -> str:
        return f"LocalizedMapping({list(self._raw)!r})"

    @property
    def raw(self) -> dict:
        return self._raw
