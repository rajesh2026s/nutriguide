"""Scope guardrails: out-of-scope retrieval and medical-question handling."""

from __future__ import annotations

import re
from typing import Literal

OUT_OF_SCOPE_MESSAGE = (
    "I don't have reliable information on that topic in the USDA Dietary "
    "Guidelines, so I can't give a grounded answer. I can help with questions "
    "about nutrition, meal planning, and dietary guidance instead."
)

CLINICAL_MESSAGE = (
    "I can't give advice about medications, dosages, or treating a medical "
    "condition — that requires a physician or registered dietitian who knows "
    "your health history. I'm happy to answer general nutrition questions "
    "based on the USDA Dietary Guidelines."
)

PERSONAL_DISCLAIMER = (
    "\n\n⚠️ *This is general information from the USDA Dietary Guidelines, not "
    "personalized medical advice. For guidance specific to your health "
    "condition, medications, or lab results, please consult a physician or "
    "registered dietitian.*"
)

# Asks for clinical decisions we must not make: medications, dosages, treatment.
_CLINICAL_RE = re.compile(
    r"\b(dos(e|es|age|ing)|medicat\w*|prescri\w*|insulin|metformin|statin|"
    r"warfarin|antibiotic\w*|chemotherapy|drug s?interact\w*)\b"
    r"|\bshould i (take|stop|start|quit)\b"
    r"|\b(treat|cure|heal)\b.{0,30}\b(disease|condition|diabetes|cancer)\b",
    re.IGNORECASE,
)

# Mentions a personal health situation: answer generally, but add a disclaimer.
_PERSONAL_RE = re.compile(
    r"\bmy (doctor|physician|condition|diagnosis|diabetes|blood pressure|"
    r"cholesterol|a1c|kidney|liver|heart|thyroid|labs?|lab (results?|values?)|"
    r"blood (sugar|test))\b"
    r"|\bi (have|was diagnosed with|suffer from|am diabetic|am pregnant)\b",
    re.IGNORECASE,
)


def classify_question(question: str) -> Literal["clinical", "personal"] | None:
    if _CLINICAL_RE.search(question):
        return "clinical"
    if _PERSONAL_RE.search(question):
        return "personal"
    return None
