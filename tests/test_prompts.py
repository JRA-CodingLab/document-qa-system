"""Tests for language-based prompt selection."""

from __future__ import annotations

from docqa.llm.prompts import get_prompts


class TestGetPrompts:
    def test_english_prompts(self):
        prompts = get_prompts("en")
        assert "system" in prompts
        assert "rag" in prompts
        assert "chat" in prompts
        assert "condense" in prompts

    def test_german_prompts(self):
        prompts = get_prompts("de")
        assert "Kontext" in prompts["system"]

    def test_fallback_to_english(self):
        prompts = get_prompts("fr")
        en_prompts = get_prompts("en")
        assert prompts["system"] == en_prompts["system"]

    def test_all_variants_have_keys(self):
        for lang in ("en", "de", "fr"):
            prompts = get_prompts(lang)
            assert set(prompts.keys()) >= {"system", "rag", "chat", "condense"}
