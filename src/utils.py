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
    if args_path.endswith(".json"):
        with open(args_path, "rb") as file:
            args = json.load(file)
    else:
        with open(args_path, "rb") as file:
            args = pickle.load(file)

    return args


def load_pkl(pkl_path: Path):
    assert str(pkl_path).endswith(".pkl")

    with open(file, "rb") as handle:
        data = pickle.load(handle)
    return data
