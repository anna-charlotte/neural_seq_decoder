# for train dataloader, do inference for the input phonemes, and store the phonemes, text, and model output, similarly to what the normal data is stored as.
from pathlib import Path

from neural_decoder.neural_decoder_trainer import getDatasetLoaders
from text2brain.models.utils import load_text2brain_model

if __name__ == "__main__":
    root_dir = Path("/data/engs-pnpl/lina4471/synthetic_data_willett2023/simple_rnn")
    data_file = Path("/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl")

    # load model from check point
    model_dir = root_dir / "model"
    model = load_text2brain_model(model_dir=model_dir)
    print(f"model.__class__.__name__ = {model.__class__.__name__}")

    # load dataloaders
    train_dl, _, _ = getDatasetLoaders(
        datasetName=data_file,
        batchSize=1,
    )

    # for synthetic data generated, also store an args file with:
    # - datapath: text data that was used as input, condition
    # (- modelpath: model that translated, generated brain signals)

    n_days = 24
    data = [
        {
            "sentenceDat": [],
            "phonemes": [],
            "phoneLens": [],
        }
        for _ in range(n_days)
    ]

    # extract data
    for batch in train_dl:

        _, text, _, _, dayIdx = batch
        # print(f"X.size() = {X.size()}")
        neural_signals = model.predict(text, dayIdx)
        # print(f"y.size() = {y.size()}")
        data[dayIdx]["sentenceDat"].append(neural_signals)
        data[dayIdx]["phonemes"].append(text)
        data[dayIdx]["phoneLens"].append(len(text))

    # store data
    file = root_dir / "data" / "pytorchTFRecords_synthetic.pkl"
    with open(file, "wb") as handle:
        print(f"Write files to: {file}")
        pickle.dump(data, handle)

    # load dat again
    with open(file, "rb") as handle:
        loaded_data = pickle.load(handle)

    train_dl = getDataLoader(
        data=loadedData,
        batch_size=16,
        shuffle=True,
        collate_fn=None,
    )
