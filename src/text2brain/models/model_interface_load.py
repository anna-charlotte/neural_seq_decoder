import json
import pickle

from text2brain.models.phoneme_image_gan import PhonemeImageGAN
from text2brain.models.vae import VAE, CondVAE
from utils import load_args


def load_t2b_gen_model(args_path, weights_path):
    args = load_args(args_path)

    assert "model_class" in args.keys(), "Model class is not given but expected."
    if args["model_class"] == "PhonemeImageGAN":
        model = PhonemeImageGAN.load_model(args_path, weights_path)
    elif args["model_class"] == "CondVAE":
        model = CondVAE.load_model(args_path, weights_path)
    elif args["model_class"] == "VAE":
        model = VAE.load_model(args_path, weights_path)

    return model
