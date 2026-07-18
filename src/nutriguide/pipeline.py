"""RAG orchestration: guardrails -> retrieval (with memory) -> generation -> verification."""

from __future__ import annotations

import logging

from nutriguide.config import Config
from nutriguide.constraints import find_violations, parse_constraints
from nutriguide.guardrails import (
    CLINICAL_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    PERSONAL_DISCLAIMER,
    classify_question,
)
from nutriguide.retriever import Passage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are NutriGuide, a dietary assistant grounded in the USDA Dietary "
    "Guidelines for Americans, 2020-2025. Answer using ONLY the context "
    "passages provided with the question; if they don't contain the answer, "
    "say you don't have that information. Do not diagnose conditions or "
    "recommend medications or dosages. Be concise and cite specific figures "
    "when the context provides them."
)

CONSTRAINT_PROMPT = (
    "\n\nSTRICT DIETARY CONSTRAINT: {constraints}. Only recommend foods that "
    "comply with this constraint, even if the context contains non-compliant "
    "examples. If the context offers only non-compliant options, say so "
    "instead of listing them."
)

CONDENSE_PROMPT = (
    "Rewrite the user's latest question as a single self-contained question, "
    'resolving references like "those", "it", or "the first one" from the '
    "conversation. Reply with the rewritten question only."
)


def _clean_history(history, max_messages: int) -> list[dict[str, str]]:
    """Normalize Gradio chat history to plain role/content messages."""
    cleaned = []
    for msg in history or []:
        role, content = msg.get("role"), msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned[-max_messages:]


class RagPipeline:
    def __init__(self, config: Config | None = None, retriever=None, generator=None):
        self.config = config or Config()
        if retriever is None:
            from nutriguide.retriever import Retriever

            retriever = Retriever.from_config(self.config)
        if generator is None:
            from nutriguide.generator import Generator

            generator = Generator(self.config.generation_model, self.config.max_new_tokens)
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str, history=None, constraints: str = "") -> str:
        question = (question or "").strip()
        if not question:
            return "Please ask a nutrition question."

        scope = classify_question(question)
        if scope == "clinical":
            return CLINICAL_MESSAGE

        history = _clean_history(history, self.config.history_turns)
        retrieval_query = self._condense(question, history) if history else question
        passages = self.retriever.search(retrieval_query, self.config.top_k)

        if not passages or passages[0].score < self.config.relevance_threshold:
            return OUT_OF_SCOPE_MESSAGE

        messages = self._build_messages(question, history, passages, constraints)
        answer = self.generator.chat(messages)
        answer = self._enforce_constraints(answer, messages, constraints)

        if scope == "personal":
            answer += PERSONAL_DISCLAIMER

        pages = ", ".join(str(p) for p in sorted({p.page_number for p in passages}))
        return f"{answer}\n\n*Source: USDA Dietary Guidelines for Americans, page(s) {pages}*"

    def _condense(self, question: str, history: list[dict[str, str]]) -> str:
        """Rewrite a follow-up into a standalone query so retrieval sees full context."""
        messages = [
            {"role": "system", "content": CONDENSE_PROMPT},
            *history,
            {"role": "user", "content": f"Latest question: {question}"},
        ]
        try:
            rewritten = self.generator.chat(
                messages, max_new_tokens=self.config.condense_max_new_tokens
            ).strip('"')
        except Exception:
            logger.exception("Query condensation failed; using the raw question")
            return question
        if not rewritten or len(rewritten) > 300:
            return question
        logger.info("Condensed %r -> %r", question, rewritten)
        return rewritten

    def _build_messages(
        self,
        question: str,
        history: list[dict[str, str]],
        passages: list[Passage],
        constraints: str,
    ) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        if constraints.strip():
            system += CONSTRAINT_PROMPT.format(constraints=constraints.strip())
        context = "\n\n".join(f"[Page {p.page_number}] {p.text}" for p in passages)
        return [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": f"Context passages:\n{context}\n\nQuestion: {question}"},
        ]

    def _enforce_constraints(
        self, answer: str, messages: list[dict[str, str]], constraints: str
    ) -> str:
        """Verify the answer against the constraint lexicon; retry once, then flag."""
        known = parse_constraints(constraints)
        if not known:
            return answer
        violations = find_violations(answer, known)
        if not violations:
            return answer

        logger.info("Constraint violations %s - regenerating", violations)
        retry = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    f"Your answer recommends {', '.join(violations)}, which violates "
                    f"the dietary constraint '{constraints}'. Rewrite the answer so it "
                    "fully complies, mentioning those foods only to advise avoiding them."
                ),
            },
        ]
        answer = self.generator.chat(retry)
        violations = find_violations(answer, known)
        if violations:
            answer += (
                f"\n\n⚠️ *Compliance check: this answer still mentions "
                f"{', '.join(violations)} — please verify it against your "
                f"'{constraints.strip()}' constraint.*"
            )
        return answer
