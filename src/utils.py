import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_args(args_path: Path) -> dict:
    if str(args_path).endswith(".json"):
        with open(args_path, "rb") as file:
            args = json.load(file)
    else:
        with open(args_path, "rb") as file:
            args = pickle.load(file)

    return args


def load_pkl(pkl_path: Path):
    assert str(pkl_path).endswith(".pkl")

    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)
    return data


def dump_args(args: dict, out_file: Path) -> None:
    if str(out_file).endswith(".json"):
        with open(out_file, "w") as file:
            json.dump(args, file, indent=4)
    else:
        with open(out_file, "wb") as file:
            pickle.dump(args, file)


def dump_pkl(args: dict, out_file: Path) -> None:
    with open(out_file, "wb") as file:
        pickle.dump(args, file)
    