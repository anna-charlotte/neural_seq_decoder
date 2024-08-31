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


def make_serializable(obj):
    """
    Recursively converts non-serializable objects to serializable formats.
    - Tensors are converted to lists.
    - NumPy arrays are converted to lists.
    - Other non-serializable objects will raise a TypeError.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return list(make_serializable(item) for item in obj)  # convert set to list
    else:
        # Attempt to serialize other types directly; will raise TypeError if not possible
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            raise TypeError(f"Object of type {type(obj).__name__} is not serializable and cannot be converted.")


def dump_json_dict(d: dict, out_file: Path) -> None:
    serializable_d = make_serializable(d)

    with open(out_file, "w") as file:
        json.dump(serializable_d, file, indent=4) 

    