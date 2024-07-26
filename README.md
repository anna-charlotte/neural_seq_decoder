## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

```pip install -e .```

# Dependencies
```
CMake >= 3.14
gcc >= 10.1
pytorch==1.13.1
```

clone speechBCI 
change name of neural_decoder to neuralDecoder in NeuralDecoder/setup.py

or clone the forked version

cd LanguageModelDecoder/runtime/server/x86
python setup.py install
cd NeuralDecoder
pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Train model: `python ./scripts/train_model.py`

