import difflib
from pathlib import Path
from typing import List, Tuple

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
    if not seq:
        return [], []

    collapsed_seq = [seq[0]]
    original_indices = [[0]]  # Track original indices for each collapsed element

    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            collapsed_seq.append(seq[i])
            original_indices.append([i])
        else:
            original_indices[-1].append(i)

    return collapsed_seq, original_indices


def remove_silence(seq, silence_placeholder):
    """Remove silence placeholders from the sequence."""
    return [item for item in seq if item != silence_placeholder]


def assign_correctness_values(pred_seq, true_seq, silence_placeholder):
    collapsed_pred_seq, original_indices = collapse_sequence(pred_seq)
    filtered_pred_seq = remove_silence(collapsed_pred_seq, silence_placeholder)
    matcher = difflib.SequenceMatcher(None, filtered_pred_seq, true_seq)

    correctness_values = ["" for _ in pred_seq]

    original_index = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                while (
                    pred_seq[original_index] == silence_placeholder
                    or original_index < len(pred_seq)
                    and pred_seq[original_index] != filtered_pred_seq[i]
                ):
                    original_index += 1
                if original_index < len(pred_seq):
                    correctness_values[original_index] = "correct"
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
                    correctness_values[original_index] = "incorrect"
                    original_index += 1

    while original_index < len(pred_seq):
        correctness_values[original_index] = (
            "incorrect" if pred_seq[original_index] != silence_placeholder else "silence"
        )
        original_index += 1

    for idx, value in enumerate(correctness_values):
        if value == "":
            correctness_values[idx] = correctness_values[idx - 1]
        if pred_seq[idx] == silence_placeholder:
            correctness_values[idx] = "silence"

    return correctness_values
