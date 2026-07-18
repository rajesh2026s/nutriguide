from nutriguide.config import Config
from nutriguide.guardrails import CLINICAL_MESSAGE, OUT_OF_SCOPE_MESSAGE, PERSONAL_DISCLAIMER
from nutriguide.pipeline import CONDENSE_PROMPT, RagPipeline
from nutriguide.retriever import Passage


class FakeRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return self.passages[:k]


class FakeGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, max_new_tokens=None):
        self.calls.append(messages)
        return self.responses.pop(0)


PASSAGES = [
    Passage(page_number=31, text="Limit sodium to 2,300 mg per day.", score=0.8),
    Passage(page_number=45, text="Adults should eat a variety of vegetables.", score=0.7),
]


def make_pipeline(responses, passages=PASSAGES):
    retriever = FakeRetriever(passages)
    generator = FakeGenerator(responses)
    pipeline = RagPipeline(config=Config(), retriever=retriever, generator=generator)
    return pipeline, retriever, generator


def test_basic_answer_has_source_footer():
    pipeline, retriever, generator = make_pipeline(["Sodium should stay under 2,300 mg."])
    answer = pipeline.answer("How much sodium per day?")
    assert "2,300 mg" in answer
    assert "page(s) 31, 45" in answer
    assert len(generator.calls) == 1  # no condensation without history


def test_out_of_scope_low_score():
    low = [Passage(page_number=1, text="irrelevant", score=0.1)]
    pipeline, _, generator = make_pipeline([], passages=low)
    assert pipeline.answer("Who won the world cup?") == OUT_OF_SCOPE_MESSAGE
    assert generator.calls == []


def test_clinical_question_refused_without_generation():
    pipeline, retriever, generator = make_pipeline([])
    assert pipeline.answer("What dose of metformin should I take?") == CLINICAL_MESSAGE
    assert generator.calls == []
    assert retriever.queries == []


def test_personal_question_gets_disclaimer():
    pipeline, _, _ = make_pipeline(["Focus on vegetables and whole grains."])
    answer = pipeline.answer("I have diabetes, what should I eat?")
    assert PERSONAL_DISCLAIMER.strip() in answer


def test_history_triggers_query_condensation():
    history = [
        {"role": "user", "content": "What are good protein sources?"},
        {"role": "assistant", "content": "Beans, lentils, poultry, and fish."},
    ]
    rewritten = "Which of beans, lentils, poultry, and fish are vegetarian-friendly?"
    pipeline, retriever, generator = make_pipeline([rewritten, "Beans and lentils."])
    answer = pipeline.answer("Which of those are vegetarian-friendly?", history=history)
    # first LLM call is the condensation prompt
    assert generator.calls[0][0]["content"] == CONDENSE_PROMPT
    # retrieval used the rewritten standalone question
    assert retriever.queries == [rewritten]
    # history is part of the final generation messages
    assert any(m["content"] == "Beans, lentils, poultry, and fish." for m in generator.calls[1])
    # the condensed standalone question (not the raw follow-up) is what gets answered
    assert rewritten in generator.calls[1][-1]["content"]
    assert "Beans and lentils." in answer


def test_constraint_violation_triggers_regeneration():
    pipeline, _, generator = make_pipeline(
        ["Try grilled chicken.", "Try beans and lentils instead."]
    )
    answer = pipeline.answer("What protein should I eat?", constraints="vegetarian")
    assert len(generator.calls) == 2
    # the retry message names the violation
    assert "chicken" in generator.calls[1][-1]["content"]
    assert "Try beans and lentils instead." in answer
    assert "Compliance check" not in answer


def test_persistent_violation_is_flagged():
    pipeline, _, _ = make_pipeline(["Eat chicken.", "Really, eat chicken."])
    answer = pipeline.answer("What protein should I eat?", constraints="vegetarian")
    assert "Compliance check" in answer
    assert "chicken" in answer


def test_empty_question():
    pipeline, _, generator = make_pipeline([])
    assert "ask a nutrition question" in pipeline.answer("  ").lower()
    assert generator.calls == []


def test_invalid_condensation_falls_back_to_stitched_query():
    history = [
        {"role": "user", "content": "What are good protein sources?"},
        {"role": "assistant", "content": "Beans and fish."},
    ]
    # the condense model answers instead of rewriting (multiline, no "?")
    pipeline, retriever, generator = make_pipeline(
        ["For children, eat:\n- Meat\n- Eggs", "Answer for kids."]
    )
    answer = pipeline.answer("What about for children?", history=history)
    # retrieval falls back to prior user turn + follow-up
    assert retriever.queries == ["What are good protein sources? What about for children?"]
    # generation still asks the raw follow-up (history resolves it)
    assert "What about for children?" in generator.calls[1][-1]["content"]
    assert "Answer for kids." in answer


def test_constraint_mentioned_in_question_is_enforced():
    pipeline, _, generator = make_pipeline(
        ["Chicken and beans are vegetarian-friendly.", "Beans are vegetarian-friendly."]
    )
    answer = pipeline.answer("Which foods are vegetarian-friendly?")
    assert len(generator.calls) == 2  # violation in first draft forced a retry
    assert "Beans are vegetarian-friendly." in answer
    assert "Compliance check" not in answer
