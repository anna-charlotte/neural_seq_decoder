import pickle
from pathlib import Path

from torch.utils.data import DataLoader

from text2brain.models.model_interface import TextToBrainInterface


def _generate_data(model: TextToBrainInterface, dl: DataLoader, n_days: int = 24) -> dict:
    data = [{"sentenceDat": [], "phonemes": [], "phoneLens": [],} for _ in range(n_days)]

    print(f"\nGenerate data ...")
    for i, batch in enumerate(dl):
        _, text, _, _, dayIdx = batch
        neural_signals = model.predict(text, dayIdx)

        data[dayIdx]["sentenceDat"].append(neural_signals[0])
        data[dayIdx]["phonemes"].append(text[0])
        data[dayIdx]["phoneLens"].append(len(text[0]))
        if i == 32:
            break
    return data


def save_synthetic_data(
    model: TextToBrainInterface,
    dl: DataLoader,
    args: dict,
    dir_path: Path,
    file_name: str = "pytorchTFRecords_synthetic.pkl",
) -> None:
    print("In store_synthetic_data() ...")

    data = _generate_data(model=model, dl=dl)

    file = dir_path / file_name
    with open(file, "wb") as handle:
        print(f"\nStore generated data to: {file}")
        pickle.dump(data, handle)

    with open(dir_path / "args", "wb") as handle:
        pickle.dump(args, handle)
    with open(dir_path / "args.json", "w") as file:
        json.dump(args, file, indent=4)
