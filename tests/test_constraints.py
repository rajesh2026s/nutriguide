from nutriguide.constraints import (
    find_mentioned_constraints,
    find_violations,
    parse_constraints,
)


def test_parse_known_constraints():
    assert parse_constraints("vegetarian, gluten free") == ["vegetarian", "gluten-free"]
    assert parse_constraints("Lactose Intolerant and low salt") == ["dairy-free", "low-sodium"]
    assert parse_constraints("keto") == []  # unknown constraints are prompt-only
    assert parse_constraints("") == []
    assert parse_constraints("vegan, vegan") == ["vegan"]


def test_violations_detected():
    text = "Good protein sources include grilled chicken, salmon, and lentils."
    assert "chicken" in find_violations(text, ["vegetarian"])
    assert "salmon" in find_violations(text, ["vegetarian"])


def test_negated_mentions_are_not_violations():
    text = "Avoid bacon and limit ham; choose fresh vegetables instead."
    assert find_violations(text, ["low-sodium"]) == []


def test_compound_food_names_are_not_violations():
    text = "Choose gluten-free bread and rice instead."
    assert find_violations(text, ["gluten-free"]) == []


def test_vegan_covers_dairy_and_eggs():
    text = "Try cheese and eggs for breakfast."
    violations = find_violations(text, ["vegan"])
    assert "cheese" in violations
    assert "egg" in violations


def test_clean_text_passes():
    text = "Lentils, beans, tofu, and quinoa are excellent choices."
    assert find_violations(text, ["vegetarian", "gluten-free"]) == []


def test_mentioned_constraints_in_questions():
    assert find_mentioned_constraints("Which of those are vegetarian-friendly?") == ["vegetarian"]
    assert find_mentioned_constraints("Is this meal suitable for vegans?") == ["vegan"]
    assert find_mentioned_constraints("What snacks are gluten-free?") == ["gluten-free"]
    # plain nutrient questions must not activate a constraint
    assert find_mentioned_constraints("What is the recommended sodium intake?") == []
    assert find_mentioned_constraints("How much dairy should I eat?") == []


def test_recommendation_after_avoid_clause_is_still_a_violation():
    text = "Avoid meat and seafood, recommending instead lean meats and poultry."
    violations = find_violations(text, ["vegetarian"])
    assert "poultry" in violations
