import pickle
import torch
from pathlib import Path
from typing import List

from text2brain.models.rnn import TextToBrainGRU
from utils import load_args


def load_text2brain_model(model_dir: Path):
    args_file = Path(model_dir / "args")
    weights_file = Path(model_dir / "modelWeights")

    assert args_file.exists()
    assert weights_file.exists()

    # with open(args_file, "rb") as handle:
    #     args = pickle.load(handle)
    args = load_args(args_file)

    if args["model_class"] == TextToBrainGRU.__name__:
        model = TextToBrainGRU(
            input_dim=args["n_input_features"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["n_output_features"],
            n_layers=args["n_layers"],
        )
    else:
        raise ValueError(f"Given model_class is not valid: {args['model_class']}")

    model.load_weights(file_path=weights_file)

    return model


def labels_to_indices(labels: torch.Tensor, classes: List[int]):
    indices = []
    classes_torch = torch.tensor(classes).to(labels.device)
    for label in labels:
        index = (classes_torch == label).nonzero(as_tuple=True)[0].item()
        indices.append(index)
    return torch.tensor(indices, dtype=torch.long, device=labels.device).view(labels.size())
