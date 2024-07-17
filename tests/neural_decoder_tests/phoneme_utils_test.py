from neural_decoder.phoneme_utils import assign_correctness_values


def test_assign_correctness_values_all_correct():
    # Additional test case to ensure correctness
    pred_seq = [1, 2, 1, 3]
    true_seq = [1, 2, 1, 3]
    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    print(correctness_values)
    assert correctness_values == ["C", "C", "C", "C"]


def test_assign_correctness_values_all_correct_with_repetitions():
    # Additional test case to ensure correctness
    pred_seq = [1, 1, 1, 1, 2]
    true_seq = [1, 2]
    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    print(correctness_values)
    assert correctness_values == ["C", "C", "C", "C", "C"]


def test_1():
    # Additional test case to ensure correctness
    pred_seq = [1, 2, 1, 8]
    true_seq = [1, 2]
    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    print(correctness_values)
    assert correctness_values == ["C", "C", "I", "I"]


def test_2():
    pred_seq = [1, 5, 2, 3, 3, 1, 0, 4, 4, 0, 5]
    true_seq = [1, 2, 3, 4, 5]

    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    assert correctness_values == [
        "C",
        "I",
        "C",
        "C",
        "C",
        "I",
        "S",
        "C",
        "C",
        "S",
        "C",
    ]


def test_3():
    # Example usage
    pred_seq = [1, 1, 2, 3, 3, 1, 1, 0, 4, 4, 0, 5, 1]
    true_seq = [1, 2, 3, 4, 5]

    correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
    assert correctness_values == [
        "C",
        "C",
        "C",
        "C",
        "C",
        "I",
        "I",
        "S",
        "C",
        "C",
        "S",
        "C",
        "I",
    ]


import torch

pred_seq = torch.tensor(
    [
        0,
        0,
        0,
        0,
        0,
        2,
        2,
        31,
        29,
        40,
        40,
        10,
        10,
        3,
        3,
        40,
        40,
        31,
        0,
        0,
        6,
        6,
        22,
        22,
        40,
        40,
        40,
        40,
        40,
        10,
        0,
        0,
        3,
        3,
        40,
        40,
        7,
        7,
        0,
        0,
        26,
        26,
        40,
        40,
        40,
        40,
        40,
        36,
        36,
        0,
        1,
        38,
        38,
        40,
        40,
        40,
        14,
        14,
        0,
        0,
        4,
        28,
        40,
        40,
        40,
        40,
        37,
        37,
        0,
        17,
        17,
        38,
        38,
        40,
        0,
        0,
        0,
        0,
        25,
        21,
        21,
        9,
        9,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
        40,
    ]
)

true_seq = torch.tensor(
    [
        2,
        31,
        40,
        10,
        3,
        40,
        31,
        6,
        22,
        40,
        10,
        3,
        40,
        7,
        26,
        40,
        36,
        1,
        38,
        40,
        14,
        4,
        28,
        40,
        37,
        17,
        28,
        38,
        40,
        25,
        21,
        9,
        40,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    device="cuda:0",
    dtype=torch.int32,
)

correctness_values = assign_correctness_values(pred_seq, true_seq, silence_placeholder=0)
print(f"correctness_values = {correctness_values}")
