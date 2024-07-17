import pickle
import tempfile
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim

from text2brain.models.phoneme_image_gan import PhonemeImageGAN, phonemes_to_signal


def test_phonemes_to_signal():
    latent_dim = 100
    phoneme_cls = [0, 1, 2, 3, 4]
    n_channels = 128
    ndf = 64
    ngf = 64
    device = "cpu"
    n_critic = 5
    clip_value = 0.01
    lr = 1e-4

    gan = PhonemeImageGAN(
        latent_dim=latent_dim,
        phoneme_cls=phoneme_cls,
        n_channels=n_channels,
        ndf=ndf,
        ngf=ngf,
        device=device,
        n_critic=n_critic,
        clip_value=clip_value,
        lr=lr,
    )
    phonemes = [2, 2, 4, 1, 2, 0, 2]
    gen_signal = phonemes_to_signal(gan, phonemes)
    print(f"gen_signal.size() = {gen_signal.size()}")
    assert gen_signal.size(0) == 1
    assert gen_signal.size(1) == len(phonemes) * 32


def test_unconditional_gan():
    latent_dim = 100
    phoneme_cls = 2
    n_channels = 128
    ndf = 64
    ngf = 64
    device = "cpu"
    n_critic = 5
    clip_value = 0.01
    lr = 1e-4

    gan = PhonemeImageGAN(
        latent_dim=latent_dim,
        phoneme_cls=phoneme_cls,
        n_channels=n_channels,
        ndf=ndf,
        ngf=ngf,
        device=device,
        n_critic=n_critic,
        clip_value=clip_value,
        lr=lr,
    )
    assert gan.conditional == False

    batch_size = 8
    X_real = torch.randn(batch_size, 32, 256).view(batch_size, n_channels, 8, 8)
    y = torch.randint(0, 10, (batch_size,))

    gan.to(device)
    X_real = X_real.to(device)
    y = y.to(device)

    errD, errG = gan((X_real, y, None, None))

    # Assertions to check the output sizes
    assert isinstance(errD, torch.Tensor), "Discriminator error should be a tensor"
    assert isinstance(errG, torch.Tensor), "Generator error should be a tensor"
    assert errD.dim() == 0, "Discriminator error should be a scalar tensor"
    assert errG.dim() == 0, "Generator error should be a scalar tensor"

    print(f"Discriminator Loss: {errD.item()}")
    print(f"Generator Loss: {errG.item()}")


def test_conditional_gan():
    latent_dim = 100
    phoneme_cls = [0, 1, 2, 3, 4]
    n_channels = 128
    ndf = 64
    ngf = 64
    device = "cpu"
    n_critic = 5
    clip_value = 0.01
    lr = 1e-4

    gan = PhonemeImageGAN(
        latent_dim=latent_dim,
        phoneme_cls=phoneme_cls,
        n_channels=n_channels,
        ndf=ndf,
        ngf=ngf,
        device=device,
        n_critic=n_critic,
        clip_value=clip_value,
        lr=lr,
    )
    assert gan.conditional == True

    batch_size = 8
    X_real = torch.randn(batch_size, 32, 256).view(batch_size, n_channels, 8, 8)
    y = torch.randint(0, max(phoneme_cls) + 1, (batch_size,))

    gan.to(device)
    X_real = X_real.to(device)
    y = y.to(device)

    errD, errG = gan((X_real, y, None, None))

    # Assertions to check the output sizes
    assert isinstance(errD, torch.Tensor), "Discriminator error should be a tensor"
    assert isinstance(errG, torch.Tensor), "Generator error should be a tensor"
    assert errD.dim() == 0, "Discriminator error should be a scalar tensor"
    assert errG.dim() == 0, "Generator error should be a scalar tensor"

    print(f"Discriminator Loss: {errD.item()}")
    print(f"Generator Loss: {errG.item()}")


def test_save_and_load_model():
    args = {
        "latent_dim": 100,
        "phoneme_cls": [0, 1, 2, 3, 4],
        "n_channels": 32,
        "ndf": 64,
        "ngf": 64,
        "device": "cpu",
        "n_critic": 5,
        "clip_value": 0.01,
        "lr": 1e-4,
    }

    gan = PhonemeImageGAN(
        latent_dim=args["latent_dim"],
        phoneme_cls=args["phoneme_cls"],
        n_channels=args["n_channels"],
        ndf=args["ndf"],
        ngf=args["ngf"],
        device=args["device"],
        n_critic=args["n_critic"],
        clip_value=args["clip_value"],
        lr=args["lr"],
    )

    with tempfile.NamedTemporaryFile(delete=False) as tmp_args, tempfile.NamedTemporaryFile(
        delete=False
    ) as tmp_weights:
        try:
            pickle.dump(args, tmp_args)
            tmp_args.flush()  # Ensure data is written to disk
            gan.save_state_dict(tmp_weights.name)

            print(tmp_weights.name)
            # Load the model
            loaded_gan = PhonemeImageGAN.load_model(tmp_args.name, tmp_weights.name)

            # Check if the loaded model's parameters match the original model's parameters
            for param_tensor in gan.state_dict():
                assert torch.equal(
                    gan.state_dict()[param_tensor], loaded_gan.state_dict()[param_tensor]
                ), f"Mismatch found in {param_tensor}"
        finally:
            import os

            os.remove(tmp_args.name)
            os.remove(tmp_weights.name)


def test_create_synthetic_phoneme_dataset():
    latent_dim = 100
    phoneme_cls = [0, 1, 2, 3, 4]
    n_channels = 128
    ndf = 64
    ngf = 64
    device = "cpu"
    n_critic = 5
    clip_value = 0.01
    lr = 1e-4

    gan = PhonemeImageGAN(
        latent_dim=latent_dim,
        phoneme_cls=phoneme_cls,
        n_channels=n_channels,
        ndf=ndf,
        ngf=ngf,
        device=device,
        n_critic=n_critic,
        clip_value=clip_value,
        lr=lr,
    )
    n_samples = 100
    label_distribution = [0.1, 0.1, 0.1, 0.35, 0.35]
    ds = gan.create_synthetic_phoneme_dataset(n_samples, label_distribution)
    print(len(ds))

    assert len(ds) == n_samples


test_create_synthetic_phoneme_dataset()
