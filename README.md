# Repository for the dissertation "Brain Data Synthesis Using Deep Generative Models"

This repository is based on a fork of the pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

Files by the authors Willet et al. are marked accordingly at the beginning of each file, or section with "# file by willet et al."

## Requirements
- python >= 3.9

## Installation

First clone the [speechBCI](https://github.com/fwillett/speechBCI) repository:
```git clone git@github.com:fwillett/speechBCI.git```

Next, you will have to got to the `NeuralDecoder/setup.py` file and change the `neural_decoder` to `neuralDecoder`.


Then follow the installation steps detailed [here](https://github.com/fwillett/speechBCI/blob/main/LanguageModelDecoder/README.md) 

And lastly:
```pip install -e .```

Now you are ready to go.

## Dependencies
```
CMake >= 3.14
gcc >= 10.1
pytorch==1.13.1
```


## Reproducing results

### Download the data

The data is publicly available and can be downloaded [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq)

### Train and evaluate generative models

There are several results to reproduce.

To train a VAE model run the `scirpts/train_t2b_vae.py` file or the according slurm script.

To train a GAN model run the `scripts/train_t2b_gan.py` file or the according slurm script.

Hyperprameters can be adjusted within the `main` section of the corresponding files.

To train a phoneme classifier on synthetic data run the `scripts/train_phoneme_classifier_synthetic.py` file or the according slurm script.
Within `main`you can sey which experiment you want to run.

To train a phoneme classifier on real data run the `scripts/train_phoneme_classifier.py` file or the according slurm script.

Training a phoneme classifier experiment will already include an evaluation. If you want to run the evaluation again after training, run the file `scripts/eval_phoneme_classifier_synthetic.py`

