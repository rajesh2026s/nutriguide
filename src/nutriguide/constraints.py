"""Dietary-constraint parsing and answer compliance checking.

Prompting alone doesn't guarantee a small model obeys "vegetarian" or
"gluten-free", so answers are verified against a lexicon of non-compliant
foods after generation (see ``RagPipeline``): a violation triggers one
corrective regeneration, and any remaining violation is flagged to the user.
"""

from __future__ import annotations

import re

_MEAT_AND_SEAFOOD = [
    "beef", "pork", "chicken", "turkey", "lamb", "veal", "bacon", "ham",
    "sausage", "salami", "pepperoni", "steak", "meat", "poultry", "gelatin",
    "fish", "salmon", "tuna", "cod", "shrimp", "crab", "lobster", "shellfish",
    "seafood", "anchovy", "sardine",
]
_DAIRY = ["milk", "cheese", "yogurt", "butter", "cream", "whey", "dairy", "ghee"]

LEXICON: dict[str, list[str]] = {
    "vegetarian": _MEAT_AND_SEAFOOD,
    "vegan": _MEAT_AND_SEAFOOD + _DAIRY + ["egg", "honey"],
    "gluten-free": [
        "wheat", "barley", "rye", "farro", "bulgur", "couscous", "seitan",
        "malt", "bread", "pasta", "noodle", "cracker", "cereal", "tortilla",
    ],
    "dairy-free": _DAIRY,
    "nut-free": [
        "peanut", "almond", "walnut", "cashew", "pecan", "pistachio",
        "hazelnut", "macadamia", "nut butter", "tree nut",
    ],
    "low-sodium": [
        "soy sauce", "cured meat", "bacon", "ham", "salami", "pickles",
        "canned soup", "processed meat", "salted",
    ],
}

_ALIASES = {
    "gluten free": "gluten-free",
    "no gluten": "gluten-free",
    "celiac": "gluten-free",
    "dairy free": "dairy-free",
    "no dairy": "dairy-free",
    "lactose-free": "dairy-free",
    "lactose free": "dairy-free",
    "lactose intolerant": "dairy-free",
    "no meat": "vegetarian",
    "meatless": "vegetarian",
    "plant-based": "vegan",
    "plant based": "vegan",
    "nut free": "nut-free",
    "no nuts": "nut-free",
    "nut allergy": "nut-free",
    "peanut allergy": "nut-free",
    "low sodium": "low-sodium",
    "low salt": "low-sodium",
    "low-salt": "low-sodium",
}

# A mention isn't a violation when it's advice to avoid the food
# ("limit bacon", "instead of cheese", "gluten-free bread").
_NEGATION_CUES = (
    "avoid", "limit", "instead of", "rather than", "without", "not ", "no ",
    "don't", "do not", "exclude", "skip", "replace", "replacing", "swap",
    "-free", "free of", "cut back", "reduce", "reducing", "allerg",
)

# ...but a recommendation verb closer to the food re-activates the violation:
# "avoid meat, recommending instead lean poultry" still recommends poultry.
_RECOMMEND_CUES = ("recommend", "choose", "opt for", "enjoy", "prefer", "try")


def parse_constraints(raw: str) -> list[str]:
    """Extract known constraint names from free-text user input.

    Unknown phrases are ignored here (they still reach the model via the
    prompt); only known constraints can be lexicon-checked.
    """
    known = []
    for part in re.split(r"[,;/]|\band\b|\n", (raw or "").lower()):
        name = re.sub(r"\s+", " ", part).strip(" .")
        name = _ALIASES.get(name, name)
        if name in LEXICON and name not in known:
            known.append(name)
    return known


_MENTION_PATTERNS = {
    "vegetarian": r"\bvegetarian\w*\b",
    "vegan": r"\bvegan\w*\b|\bplant[- ]based\b",
    "gluten-free": r"\bgluten[- ]free\b|\bavoid\w* gluten\b|\bwithout gluten\b|\bceliac\b",
    "dairy-free": r"\bdairy[- ]free\b|\blactose\b",
    "nut-free": r"\bnut[- ]free\b|\bnut allerg\w*\b|\bpeanut allerg\w*\b",
    "low-sodium": r"\blow[- ]sodium\b|\blow[- ]salt\b",
}


def find_mentioned_constraints(text: str) -> list[str]:
    """Known constraints referenced in free text, e.g. a question asking
    "which of those are vegetarian-friendly?" implies the vegetarian lexicon."""
    return [
        name
        for name, pattern in _MENTION_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]


def _is_negated(text: str, match_start: int) -> bool:
    sentence_start = max(
        text.rfind(ch, 0, match_start) for ch in ".!?\n"
    )
    window = text[max(sentence_start + 1, match_start - 90) : match_start].lower()
    nearest_negation = max(window.rfind(cue) for cue in _NEGATION_CUES)
    nearest_recommend = max(window.rfind(cue) for cue in _RECOMMEND_CUES)
    return nearest_negation != -1 and nearest_negation >= nearest_recommend


def find_violations(text: str, constraint_names: list[str]) -> list[str]:
    """Return forbidden foods the text recommends (mentions without negation)."""
    violations = []
    for name in constraint_names:
        for term in LEXICON[name]:
            pattern = re.compile(rf"\b{re.escape(term)}(e?s)?\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                if not _is_negated(text, match.start()):
                    if term not in violations:
                        violations.append(term)
                    break
    return violations
