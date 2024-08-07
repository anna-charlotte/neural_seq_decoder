import json
import pickle
from pathlib import Path

from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.models.vae import VAE, CondVAE
from utils import load_args


def load_t2b_gen_model(weights_path, args_path=None):
    """ 
    Load a specified model with given weights and configuration.

    This function loads a T2B model (one of: PhonemeImageGAN, CondVAE, VAE) from specified 
    weights and configuration files. If `args_path` is not provided, it assumes a JSON file 
    with the same name as the weights in the same directory.
    """
    
    if args_path is None:
        args_path = Path(weights_path).parent / "args.json"

    args = load_args(args_path)

    assert "model_class" in args.keys(), "Model class is not given but expected."
    if args["model_class"] == "PhonemeImageGAN":
        model = PhonemeImageGAN.load_model(args_path, weights_path)
    elif args["model_class"] == "CondVAE":
        model = CondVAE.load_model(args_path, weights_path)
    elif args["model_class"] == "VAE":
        model = VAE.load_model(args_path, weights_path)

    return model
