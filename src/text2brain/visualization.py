
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from neural_decoder.neural_decoder_trainer import get_dataset_loaders
from neural_decoder.phoneme_utils import ROOT_DIR

channel_order_1 = [
    62, 51, 43, 35, 94, 87, 79, 78,
    60, 53, 41, 33, 95, 86, 77, 76,
    63, 54, 47, 44, 93, 84, 75, 74,
    58, 55, 48, 40, 92, 85, 73, 72,
    59, 45, 46, 38, 91, 82, 71, 70,
    61, 49, 42, 36, 90, 83, 69, 68,
    56, 52, 39, 34, 89, 81, 67, 66,
    57, 50, 37, 32, 88, 80, 65, 64
]
channel_order_2 = [
    125, 126, 112, 103, 31, 28, 11, 8,
    123, 124, 110, 102, 29, 26, 9, 5,
    121, 122, 109, 101, 27, 19, 18, 4,
    119, 120, 108, 100, 25, 15, 12, 6,
    117, 118, 107, 99, 23, 13, 10, 3,
    115, 116, 106, 97, 21, 20, 7, 2,
    113, 114, 105, 98, 17, 24, 14, 0,
    127, 111, 104, 96, 30, 22, 16, 1
]


def plot_brain_signal_animation(signal: torch.Tensor, save_path: Path) -> None:
    
    assert len(signal.size()) == 2
    assert signal.size(1) == 256
    print(signal.size())
    
    reshaped_signal = signal.view(4, -1, 8, 8)
    img_list = []

    for i in range(reshaped_signal.size(1)):
        imgs = []
        for j, img in enumerate(reshaped_signal[:, i, :, :]):
            # if j == 0 or j == 2:
            #     imgs.append(img[channel_order_1])
            # if j == 1 or j == 3:
            #     imgs.append(img[channel_order_2])
            imgs.append(img)

        img_list.append(imgs)

        if i == 20:
            break

    fig, axs = plt.subplots(2, 2)
    ims = []
    cbs = []

    for ax, img in zip(axs.flatten(), img_list[0]):
        im = ax.imshow(img, cmap='viridis')
        ims.append(im)
        cb = fig.colorbar(im, ax=ax) 
        cbs.append(cb)
    
    titles = ["Signal 1 (6v)", "Signal 1 (6v)", "Spike power 1", "Spike power 1"]

    def animate(i):
        for j, (im, img, ax) in enumerate(zip(ims, img_list[i], axs.flatten())):
            cbs[j].remove() 
            cb = fig.colorbar(im, ax=ax)
            cbs[j] = cb
            ax.set_title(titles[j])

            im.set_data(img) 

        fig.suptitle(f'Frame {i}')

        return ax

    ani = animation.FuncAnimation(fig, animate, frames=len(img_list), interval=200, repeat=False)
    ani.save(save_path, writer='imagemagick', fps=5)


if __name__ == "__main__":
    dataset_path = "/data/engs-pnpl/lina4471/willett2023/competitionData/pytorchTFRecords.pkl"
    print(ROOT_DIR)
    save_path = ROOT_DIR / "plots"/ "data_visualization" / "animation.gif" 
    print(save_path)

    train_dl, test_dl, loaded_data = get_dataset_loaders(dataset_path, batch_size=1)

    for i, batch in enumerate(train_dl):
        X, y, _, _, _ = batch
        for signal in X:
            plot_brain_signal_animation(signal=signal, save_path=save_path)
            break
        break

