import pickle
import tempfile
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim

from text2brain.models.vae import VAE


def test_save_and_load_VAE():
    args = {"latent_dim": 10, "input_shape": (1, 256, 32), "device": "cpu"}

    model = VAE(latent_dim=args["latent_dim"], input_shape=args["input_shape"]).to(args["device"])

    with tempfile.NamedTemporaryFile(delete=False) as tmp_args, tempfile.NamedTemporaryFile(
        delete=False
    ) as tmp_weights:
        try:
            # save model args
            pickle.dump(args, tmp_args)
            tmp_args.flush()  # Ensure data is written to disk

            # save model weights
            model.save_state_dict(tmp_weights.name)
            tmp_weights.flush()  # Ensure data is written to disk

            # load the model
            loaded_model = VAE.load_model(Path(tmp_args.name), Path(tmp_weights.name))

            # check if the loaded model's parameters match the original model's parameters
            for param_tensor in model.state_dict():
                assert torch.equal(
                    model.state_dict()[param_tensor], loaded_model.state_dict()[param_tensor]
                ), f"Mismatch found in {param_tensor}"
        finally:
            import os

            os.remove(tmp_args.name)
            os.remove(tmp_weights.name)
