"""Tests for language detection."""

from __future__ import annotations

from docqa.vectordb.language import detect_language


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello, how are you?") == "en"

    def test_german(self):
        assert detect_language("Die Katze sitzt auf dem Tisch") == "de"

    def test_short_text_defaults_english(self):
        assert detect_language("OK") == "en"

    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_mixed_german_majority(self):
        text = "Das ist ein Test mit some English words und die Katze"
        assert detect_language(text) == "de"
