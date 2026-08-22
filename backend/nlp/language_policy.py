"""Strict Chinese/English language policy for the interactive UI."""

from __future__ import annotations

import unicodedata


SUPPORTED_RESPONSE_LANGUAGES = ("zh-CN", "en")


def normalize_requested_language(candidate: str | None) -> str:
    """Normalize an explicit output locale without inspecting question text."""

    normalized = str(candidate or "").strip().casefold().replace("_", "-")
    if normalized.startswith("zh"):
        return "zh-CN"
    if normalized.startswith("en"):
        return "en"
    return "en"


def normalize_supported_language(
    candidate: str | None,
    *,
    text: str,
) -> str:
    """Return only ``zh-CN`` or ``en``, with the user text authoritative."""

    detected = detect_supported_text_language(text)
    if detected is not None:
        return detected
    normalized = str(candidate or "").strip().casefold().replace("_", "-")
    if normalized.startswith("zh"):
        return "zh-CN"
    if normalized.startswith("en"):
        return "en"
    # Unsupported or uncertain input is handled in English.  Most
    # importantly, labels such as ja/ar/de can never reach generation.
    return "en"


def detect_supported_text_language(text: str) -> str | None:
    """Identify Chinese/English by writing system and reject Japanese kana."""

    value = str(text or "")
    if any(_is_japanese_kana(character) for character in value):
        return None
    han_count = sum(_is_han(character) for character in value)
    latin_count = sum(_is_latin_letter(character) for character in value)
    if han_count:
        return "zh-CN"
    if latin_count:
        return "en"
    return None


def output_matches_supported_language(text: str, language: str) -> bool:
    expected = normalize_supported_language(language, text="")
    return detect_supported_text_language(text) == expected


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _is_japanese_kana(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x309F
        or 0x30A0 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xFF66 <= codepoint <= 0xFF9D
    )


def _is_latin_letter(character: str) -> bool:
    if not character.isalpha():
        return False
    try:
        return "LATIN" in unicodedata.name(character)
    except ValueError:
        return False
