"""Lightweight language detection (English vs German)."""

from __future__ import annotations

_GERMAN_INDICATORS = frozenset(
    [
        "der", "die", "das", "und", "ist", "sind", "ein", "eine",
        "für", "mit", "auf", "nicht", "ich", "du", "wir", "sie",
        "kann", "wird", "haben", "werden", "über", "nach",
    ]
)


def detect_language(text: str) -> str:
    """Return ``"en"`` or ``"de"`` for the dominant language in *text*.

    Uses ``langdetect`` when available and falls back to a simple
    keyword heuristic otherwise.  Defaults to ``"en"`` on any error.
    """
    if not text or not text.strip():
        return "en"

    # Try the proper library first
    try:
        from langdetect import detect

        result = detect(text)
        return "de" if result == "de" else "en"
    except ImportError:
        pass
    except Exception:
        return "en"

    # Fallback heuristic
    try:
        words = text.lower().split()
        if not words:
            return "en"
        german_count = sum(1 for w in words if w in _GERMAN_INDICATORS)
        if german_count >= 2 or (len(words) > 0 and german_count / len(words) > 0.2):
            return "de"
        return "en"
    except Exception:
        return "en"
