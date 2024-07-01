from neural_decoder.phoneme_utils import assign_correctness_values


def test_assign_correctness_values_all_correct():
    # Additional test case to ensure correctness
    pred_seq = [1, 2, 1, 3]
    true_seq = [1, 2, 1, 3]
    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    print(correctness_values)
    assert correctness_values == ["correct", "correct", "correct", "correct"]


def test_assign_correctness_values_all_correct_with_repetitions():
    # Additional test case to ensure correctness
    pred_seq = [1, 1, 1, 1, 2]
    true_seq = [1, 2]
    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    print(correctness_values)
    assert correctness_values == ["correct", "correct", "correct", "correct", "correct"]


def test_1():
    # Additional test case to ensure correctness
    pred_seq = [1, 2, 1, 8]
    true_seq = [1, 2]
    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    print(correctness_values)
    assert correctness_values == ["correct", "correct", "incorrect", "incorrect"]


def test_2():
    pred_seq = [1, 5, 2, 3, 3, 1, 0, 4, 4, 0, 5]
    true_seq = [1, 2, 3, 4, 5]

    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    assert correctness_values == [
        "correct",
        "incorrect",
        "correct",
        "correct",
        "correct",
        "incorrect",
        "silence",
        "correct",
        "correct",
        "silence",
        "correct",
    ]


def test_3():

    # Example usage
    pred_seq = [1, 1, 2, 3, 3, 1, 1, 0, 4, 4, 0, 5, 1]
    true_seq = [1, 2, 3, 4, 5]

    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    assert correctness_values == [
        "correct",
        "correct",
        "correct",
        "correct",
        "correct",
        "incorrect",
        "incorrect",
        "silence",
        "correct",
        "correct",
        "silence",
        "correct",
        "incorrect",
    ]


# Example usage
pred_seq = [1, 1, 2, 3, 3, 1, 1, 0, 4, 4, 0, 5, 1]
true_seq = [1, 2, 3, 4, 5]

correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
print(correctness_values)
assert correctness_values == [
    "correct",
    "correct",
    "correct",
    "correct",
    "correct",
    "incorrect",
    "incorrect",
    "silence",
    "correct",
    "correct",
    "silence",
    "correct",
    "incorrect",
]
