import pickle
from pathlib import Path

from text2brain.models.rnn import TextToBrainGRU


def load_text2brain_model(model_dir: Path):
    args_file = Path(model_dir / "args")
    weights_file = Path(model_dir / "modelWeights")

    assert args_file.exists()
    assert weights_file.exists()

    with open(args_file, "rb") as handle:
        args = pickle.load(handle)

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
