import difflib
from pathlib import Path
from typing import List, Tuple
import torch

ROOT_DIR = Path(__file__).parent.parent.parent


PHONE_DEF = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
PHONE_DEF_SIL = PHONE_DEF + ["SIL"]


def phone_to_id(p: str):
    return PHONE_DEF_SIL.index(p)


def id_to_phone(idx: int):
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
