"""
adaptation.py — Adaptive Interaction Layer for NutriGuide
Derives a response detail level from session interaction patterns
(AI-based adaptive HCI). Stateless by design: recomputed from the
Gradio-managed `history` each turn, since RagPipeline is a single
instance shared across concurrent user sessions.
"""

from __future__ import annotations

from enum import Enum


class DetailLevel(str, Enum):
    CONCISE = "concise"
    STANDARD = "standard"
    DETAILED = "detailed"


_LADDER = [DetailLevel.CONCISE, DetailLevel.STANDARD, DetailLevel.DETAILED]
SHORT_MSG_WORD_THRESHOLD = 6
FOLLOWUP_KEYWORDS = ("more", "what about", "why", "explain", "elaborate", "detail")


def _shift(level: DetailLevel, steps: int) -> DetailLevel:
    idx = _LADDER.index(level)
    idx = max(0, min(len(_LADDER) - 1, idx + steps))
    return _LADDER[idx]


def compute_detail_level(history: list[dict[str, str]], question: str) -> DetailLevel:
    """Replays the user's turns in `history` plus the current `question`
    to derive the current adaptive detail level. Pure function - safe to
    call independently from multiple places without shared state."""
    level = DetailLevel.STANDARD
    followup_streak = 0
    short_streak = 0

    user_turns = [m["content"] for m in history if m.get("role") == "user"]
    user_turns.append(question)

    for msg in user_turns:
        words = len(msg.split())
        is_short = words <= SHORT_MSG_WORD_THRESHOLD
        is_followup = any(kw in msg.lower() for kw in FOLLOWUP_KEYWORDS)

        followup_streak = followup_streak + 1 if is_followup else 0
        short_streak = short_streak + 1 if is_short else 0

        if followup_streak >= 2:
            level = _shift(level, +1)
            followup_streak = 0
        if short_streak >= 3:
            level = _shift(level, -1)
            short_streak = 0

    return level


def detail_instruction(level: DetailLevel) -> str:
    """Prompt-injection text matching the current detail level."""
    return {
        DetailLevel.CONCISE: (
            "Answer in 1-2 short sentences. No elaboration unless explicitly asked."
        ),
        DetailLevel.STANDARD: (
            "Answer in 2-4 sentences with a balanced level of explanation."
        ),
        DetailLevel.DETAILED: (
            "Answer thoroughly, with context, examples, and relevant caveats "
            "from the source material."
        ),
    }[level]