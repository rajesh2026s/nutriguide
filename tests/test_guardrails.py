from nutriguide.guardrails import classify_question


def test_clinical_questions():
    assert classify_question("What dose of metformin should I take?") == "clinical"
    assert classify_question("Should I stop taking my blood pressure medication?") == "clinical"
    assert classify_question("Can diet cure my disease?") == "clinical"


def test_personal_questions():
    assert classify_question("I have diabetes, what should I eat?") == "personal"
    assert classify_question("My cholesterol is high, which foods help?") == "personal"


def test_general_questions():
    assert classify_question("How much fiber should adults eat per day?") is None
    assert classify_question("What are good sources of vitamin D?") is None
