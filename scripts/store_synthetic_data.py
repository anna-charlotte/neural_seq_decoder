# load model from check point

# load dataloaders
# for train dataloader, do inference for the input phonemes, and store the phonemes, text, and model output, similarly to what the normal data is stored as.
from pathlib import Path

from text2brain.models.utils import load_text2brain_model

if __name__ == "__main__":
    directory = Path("/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn")

    model_dir = directory / "model"
    model = load_text2brain_model(model_dir=model_dir)
