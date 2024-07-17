import difflib
from pathlib import Path
from typing import List, Tuple

import torch

ROOT_DIR = Path(__file__).parent.parent.parent


# fmt: off
PHONE_DEF = [
    "AA", "AE", "AH", "AO", "AW", "AY", 
    "B", "CH", "D", "DH", "EH", "ER", 
    "EY", "F", "G", "HH", "IH", "IY",
     "JH", "K", "L", "M", "N", "NG",
    "OW", "OY", "P", "R", "S", "SH", 
    "T", "TH", "UH", "UW", "V", "W", 
    "Y", "Z", "ZH"
]
PHONE_DEF_SIL = PHONE_DEF + ["SIL"]

CHANNEL_ORDER = [
    62, 51, 43, 35, 94, 87, 79, 78,
    60, 53, 41, 33, 95, 86, 77, 76,
    63, 54, 47, 44, 93, 84, 75, 74,
    58, 55, 48, 40, 92, 85, 73, 72,
    59, 45, 46, 38, 91, 82, 71, 70,
    61, 49, 42, 36, 90, 83, 69, 68,
    56, 52, 39, 34, 89, 81, 67, 66,
    57, 50, 37, 32, 88, 80, 65, 64,
    125, 126, 112, 103, 31, 28, 11, 8,
    123, 124, 110, 102, 29, 26, 9, 5,
    121, 122, 109, 101, 27, 19, 18, 4,
    119, 120, 108, 100, 25, 15, 12, 6,
    117, 118, 107, 99, 23, 13, 10, 3,
    115, 116, 106, 97, 21, 20, 7, 2,
    113, 114, 105, 98, 17, 24, 14, 0,
    127, 111, 104, 96, 30, 22, 16, 1,
]
# fmt: on

DISTANCE_METRICS = ["frobenius", "cosine_sim", "manhattan", "mse"]


def reorder_neural_window(tensor: torch.Tensor) -> torch.Tensor:
    print(f"tensor.size() = {tensor.size()}")
    assert tensor.size() == (256, 32), f"size should be (256, 32) but is: {tensor.size()}"
    reordered_tensor = torch.empty_like(tensor)
    reordered_tensor[:128] = tensor[CHANNEL_ORDER]
    reordered_tensor[128:] = tensor[[i + 128 for i in CHANNEL_ORDER]]
    return reordered_tensor


def phone_to_id(p: str) -> int:
    return PHONE_DEF_SIL.index(p)


def id_to_phone(idx: int) -> str:
    assert idx in range(0, len(PHONE_DEF_SIL))
    return PHONE_DEF_SIL[idx]


def collapse_sequence(seq: List[int]) -> Tuple[List[int], List[List[int]]]:
    """Collapse consecutive identical phonemes in the sequence."""

    collapsed_seq = [seq[0]]
    original_indices = [[0]]

    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            collapsed_seq.append(seq[i])
            original_indices.append([i])
        else:
            original_indices[-1].append(i)

    return collapsed_seq, original_indices


def remove_silence(seq: List[int], silence_placeholder: int) -> List[int]:
    """Remove silence placeholders from the sequence."""
    return [item for item in seq if item != silence_placeholder]


def assign_correctness_values(
    pred_seq: List[int], true_seq: List[int], silence_placeholder: int
) -> List[str]:
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.tolist()
    if isinstance(true_seq, torch.Tensor):
        true_seq = true_seq.tolist()

    collapsed_pred_seq, _ = collapse_sequence(pred_seq)
    filtered_pred_seq = remove_silence(collapsed_pred_seq, silence_placeholder)
    matcher = difflib.SequenceMatcher(None, filtered_pred_seq, true_seq)

    correctness_values = ["" for _ in pred_seq]

    original_index = 0
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                while (
                    pred_seq[original_index] == silence_placeholder
                    or original_index < len(pred_seq)
                    and pred_seq[original_index] != filtered_pred_seq[i]
                ):
                    original_index += 1
                if original_index < len(pred_seq):
                    correctness_values[original_index] = "C"
                    original_index += 1
        elif tag in ("replace", "delete", "insert"):
            for i in range(i1, i2):
                while (
                    pred_seq[original_index] == silence_placeholder
                    or original_index < len(pred_seq)
                    and pred_seq[original_index] != filtered_pred_seq[i]
                ):
                    original_index += 1
                if original_index < len(pred_seq):
                    correctness_values[original_index] = "I"
                    original_index += 1

    while original_index < len(pred_seq):
        correctness_values[original_index] = "I" if pred_seq[original_index] != silence_placeholder else "S"
        original_index += 1

    for idx, value in enumerate(correctness_values):
        if value == "":
            correctness_values[idx] = correctness_values[idx - 1]
        if pred_seq[idx] == silence_placeholder:
            correctness_values[idx] = "S"

    return correctness_values

