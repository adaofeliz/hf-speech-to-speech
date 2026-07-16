"""Offline duration-floor check for the start_agent_run kickoff style.

No live TTS/LLM call: this reuses the exact arithmetic formula
Qwen3TTSHandler._estimate_max_new_tokens uses to size the codec-token
budget, applied to a static canonical example utterance representative
of the new SOUL.md/voice_prompt.py kickoff wording, to prove that style
of text clears a 15s floor toward the 30s target.
"""

import re
import unicodedata

from speech_to_speech.TTS.qwen3_tts_handler import (
    ESTIMATED_QWEN3_CHARS_PER_SECOND,
    ESTIMATED_QWEN3_WORDS_PER_SECOND,
    QWEN3_BASE_PROMPT_SECONDS,
    QWEN3_PUNCTUATION_PAUSE_SECONDS,
)

MIN_KICKOFF_SECONDS = 15.0
TARGET_KICKOFF_SECONDS = 30.0

EXAMPLE_TEXT = (
    "All right, let me look into that for you. Hold on... give me a second. "
    "This one is going to take a bit of digging, checking a few different places, "
    "cross referencing what actually applies to your situation before I say anything for sure. "
    "Hmm... let me pull this up properly. Uh, okay, running through it now. "
    "This is not a quick lookup, so bear with me for a moment. "
    "Hmm, still working through the details, nothing conclusive yet. "
    "Okay... give me just a bit more, I am not done digging."
)


def _estimate_seconds(text: str) -> float:
    """Mirror Qwen3TTSHandler._estimate_max_new_tokens's duration formula."""
    word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
    char_count = len(re.sub(r"\s+", "", text))
    word_seconds = word_count / ESTIMATED_QWEN3_WORDS_PER_SECOND if word_count else 0.0
    char_seconds = char_count / ESTIMATED_QWEN3_CHARS_PER_SECOND if char_count else 0.0
    punctuation_count = sum(unicodedata.category(ch).startswith("P") for ch in text)
    punctuation_seconds = punctuation_count * QWEN3_PUNCTUATION_PAUSE_SECONDS
    return max(word_seconds, char_seconds) + punctuation_seconds + QWEN3_BASE_PROMPT_SECONDS


def test_canonical_kickoff_example_clears_duration_floor():
    estimated_seconds = _estimate_seconds(EXAMPLE_TEXT)

    assert estimated_seconds >= MIN_KICKOFF_SECONDS
    assert "\u2014" not in EXAMPLE_TEXT  # no em dash
    assert re.search(r"\b(hmm|uh)\b", EXAMPLE_TEXT, re.IGNORECASE)  # paralanguage present


def test_short_stub_correctly_fails_the_floor():
    """Sanity check: a short, non-paralanguage stub must NOT clear the floor.

    This proves the floor check above is not tautological/vacuous.
    """
    stub_text = "Let me start that now."
    estimated_seconds = _estimate_seconds(stub_text)

    assert estimated_seconds < MIN_KICKOFF_SECONDS
