"""RAG orchestration: guardrails -> retrieval (with memory) -> generation -> verification."""

from __future__ import annotations

import logging

from nutriguide.config import Config
from nutriguide.constraints import (
    find_mentioned_constraints,
    find_violations,
    parse_constraints,
)
from nutriguide.guardrails import (
    CLINICAL_MESSAGE,
    GREETING_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    PERSONAL_DISCLAIMER,
    classify_question,
    is_greeting,
)
from nutriguide.retriever import Passage
from nutriguide.adaptation import compute_detail_level, detail_instruction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are NutriGuide, a dietary assistant grounded in the USDA Dietary "
    "Guidelines for Americans, 2020-2025. Ground every nutrition fact in the "
    "context passages provided with the latest question; if they don't "
    "contain the answer, say you don't have that information. Refer to your "
    "source as 'the Dietary Guidelines', never as 'the context'. Use the "
    "earlier conversation to understand what the user refers to - when they "
    "say 'those' or 'them', they mean items from your earlier answers, so "
    "answer about exactly those items. Do not diagnose conditions or "
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
    'conversation. Example: after an answer listing spinach, kale, and '
    'carrots, "which of these are high in vitamin A?" becomes "Which of '
    'spinach, kale, and carrots are high in vitamin A?". If the latest '
    "question is already self-contained, return it unchanged. Reply with the "
    "rewritten question only - do not answer it."
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

        if is_greeting(question):
            return GREETING_MESSAGE

        scope = classify_question(question)
        if scope == "clinical":
            return CLINICAL_MESSAGE

        history = _clean_history(history, self.config.history_turns)
        # the condensed standalone question drives BOTH retrieval and generation:
        # a small model answers "Which of X, Y, Z ..." far more reliably than a
        # follow-up whose references it must resolve from history itself
        standalone = self._condense(question, history) if history else question
        if standalone is None:
            # condensation produced garbage; retrieve on prior turn + question
            # and let the generator resolve the follow-up from history instead
            prior_user = [m["content"] for m in history if m["role"] == "user"]
            retrieval_query = " ".join(prior_user[-1:] + [question])
            standalone = question
        else:
            retrieval_query = standalone
        passages = self.retriever.search(retrieval_query, self.config.top_k)

        if not passages or passages[0].score < self.config.relevance_threshold:
            return OUT_OF_SCOPE_MESSAGE

        level = compute_detail_level(history, question)
        messages = self._build_messages(standalone, history, passages, constraints, level)
        answer = self.generator.chat(messages)
        answer = self._enforce_constraints(answer, messages, standalone, constraints)

        if scope == "personal":
            answer += PERSONAL_DISCLAIMER

        pages = ", ".join(str(p) for p in sorted({p.page_number for p in passages}))
        return f"{answer}\n\n*Source: USDA Dietary Guidelines for Americans, page(s) {pages}*"

    def _condense(self, question: str, history: list[dict[str, str]]) -> str | None:
        """Rewrite a follow-up into a standalone question, or None if the rewrite
        is unusable (small models sometimes answer instead of rewriting)."""
        messages = [
            {"role": "system", "content": CONDENSE_PROMPT},
            *history,
            {"role": "user", "content": f"Latest question: {question}"},
        ]
        try:
            rewritten = self.generator.chat(
                messages, max_new_tokens=self.config.condense_max_new_tokens
            ).strip().strip('"')
        except Exception:
            logger.exception("Query condensation failed; using the raw question")
            return None
        # a valid rewrite is one line, question-shaped, and reasonably short
        if "\n" in rewritten or not rewritten.endswith("?") or not 0 < len(rewritten) <= 300:
            logger.warning("Discarding invalid condensation %r", rewritten)
            return None
        logger.info("Condensed %r -> %r", question, rewritten)
        return rewritten

    def _build_messages(
        self,
        question: str,
        history: list[dict[str, str]],
        passages: list[Passage],
        constraints: str,
        detail_level=None,
    ) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        if detail_level is not None:
            system += "\n\n" + detail_instruction(detail_level)
        if constraints.strip():
            system += CONSTRAINT_PROMPT.format(constraints=constraints.strip())
        context = "\n\n".join(f"[Page {p.page_number}] {p.text}" for p in passages)
        return [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": f"Context passages:\n{context}\n\nQuestion: {question}"},
        ]

    def _enforce_constraints(
        self, answer: str, messages: list[dict[str, str]], question: str, constraints: str
    ) -> str:
        """Verify the answer against the constraint lexicon; retry once, then flag."""
        known = parse_constraints(constraints)
        # a constraint named in the question itself ("...vegetarian-friendly?")
        # is verified too, even when the constraints field is empty
        for name in find_mentioned_constraints(question):
            if name not in known:
                known.append(name)
        if not known:
            return answer
        label = constraints.strip() or ", ".join(known)
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
                    f"the dietary constraint '{label}'. Rewrite the answer so it "
                    "fully complies, mentioning those foods only to advise avoiding "
                    "them. Reply with only the corrected answer - do not apologize or "
                    "refer to this correction."
                ),
            },
        ]
        answer = self.generator.chat(retry)
        violations = find_violations(answer, known)
        if violations:
            answer += (
                f"\n\n⚠️ *Compliance check: this answer still mentions "
                f"{', '.join(violations)} — please verify it against your "
                f"'{label}' constraint.*"
            )
        return answer
