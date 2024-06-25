import pickle
from pathlib import Path

from neural_decoder.dataset import _padding
from neural_decoder.neural_decoder_trainer import getDataLoader, getDatasetLoaders
from text2brain.models.utils import load_text2brain_model
from text2brain.synthetic_data import save_synthetic_data

if __name__ == "__main__":
    root_dir = Path("/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn")
    input_data_file = Path("/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl")

    # load model from check point
    model_dir = root_dir / "model"
    model = load_text2brain_model(model_dir=model_dir)

    # load dataloaders
    train_dl, _, _ = getDatasetLoaders(
        datasetName=input_data_file,
        batchSize=1,
    )

    file_name = "pytorchTFRecords_synthetic.pkl"
    save_synthetic_data(
        model=model,
        dl=train_dl,
        dir_path=root_dir / "synthetic_data",
        file_name=file_name,
        args={"input_data_file": input_data_file},
    )

    # for synthetic data generated, also store an args file with:
    # - datapath: text data that was used as input, condition
    # (- modelpath: model that translated, generated brain signals)

    file = root_dir / "synthetic_data" / "pytorchTFRecords_synthetic.pkl"
    # load dat again
    print(f"\nLoad stored generated data from: {file}")
    with open(file, "rb") as handle:
        loaded_data = pickle.load(handle)

    train_dl = getDataLoader(
        data=loaded_data,
        batch_size=16,
        shuffle=True,
        collate_fn=_padding,
    )
    for i, batch in enumerate(train_dl):
        print()
        y, text, y_len, X_len, dayIdx = batch
        print(f"text.size() = {text.size()}")
        print(f"y.size() = {y.size()}")
        print(f"type(text) = {type(text)}")
        neural_signals = model.predict(text, dayIdx)
