from pathlib import Path
from neural_decoder.phoneme_utils import ROOT_DIR
import json
from evaluation import compute_auroc_with_stderr, compute_cross_correlation
import numpy as np
import seaborn as sns
from datetime import datetime
from text2brain.visualization import plot_aurocs_with_error_bars
import re
from text2brain.models.model_interface_load import load_t2b_gen_model
from neural_decoder.neural_decoder_trainer import get_data_loader
from torchvision import transforms
from neural_decoder.transforms import (
    AddOneDimensionTransform,
    ReorderChannelTransform,
    SoftsignTransform,
    TransposeTransform,
)
import matplotlib.pyplot as plt
import torch
import random
from data.dataset import PhonemeDataset
from utils import load_pkl, set_seeds
from data.augmentations import GaussianSmoothing


def evaluate_vae_latent_dim_experiment(
    model_dir: Path, 
    file_name_pattern: str,
    pattern_latent_dims: list, 
    pattern_seeds: list,
    plot_dir: Path,
    n_syn_samples: int, 
    bootstrap_iters: int = 1_000,
    metric: str = "auroc",
) -> None:
    assert "{latent_dim}" in file_name_pattern, "'file_name_pattern' must contain '{latent_dim}' placeholder"

    plot_dir.mkdir(parents=True, exist_ok=True)

    aurocs = []
    sems = []
    syn_aurocs = []
    syn_sems = []
    x_labels = []

    for latent_dim in pattern_latent_dims:
        print(f"latent_dim = {latent_dim}")
        seed_aurocs = []
        syn_seed_aurocs = []

        for pattern_seed in pattern_seeds:
            sub_dir = model_dir / file_name_pattern.format(latent_dim=latent_dim, seed=pattern_seed) / "phoneme_classifier"
            for out_dir in sub_dir.iterdir():
                if out_dir.is_dir():
                    if f"syn_n_samples_{n_syn_samples}" in str(out_dir):

                        with open(out_dir / "ouput.json", 'r') as f:
                            output = json.load(f)
                        
                        auroc, _ = compute_auroc_with_stderr(
                            y_true=output["y_true"], 
                            y_pred=output["y_pred"], 
                            n_iters=bootstrap_iters, 
                        )
                        seed_aurocs.append(auroc)

                        syn_auroc, _ = compute_auroc_with_stderr(
                            y_true=output["y_true_syn"], 
                            y_pred=output["y_pred_syn"], 
                            n_iters=bootstrap_iters, 
                        )
                        syn_seed_aurocs.append(syn_auroc)

        
        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=len(seed_aurocs), replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        auroc = np.mean(seed_aurocs)
        sem = np.std(bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        aurocs.append(auroc)
        sems.append(sem)

        syn_bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(syn_seed_aurocs, size=len(seed_aurocs), replace=True)
            syn_bootstrapped_aurocs.append(np.mean(sample))

        syn_auroc = np.mean(syn_seed_aurocs)
        syn_sem = np.std(syn_bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        syn_aurocs.append(syn_auroc)
        syn_sems.append(syn_sem)

        x_labels.append(latent_dim)

        colors = sns.color_palette("Set2", 3)[2]

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        plot_aurocs_with_error_bars(
            aurocs=aurocs, 
            errs=sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_REAL__syn_n_samples_{n_syn_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions on real data",
            colors=colors,
        )
        plot_aurocs_with_error_bars(
            aurocs=aurocs, 
            errs=sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_REAL_with_annotations__syn_n_samples_{n_syn_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions on real data",
            colors=colors,
            add_annotations=True
        )


        plot_aurocs_with_error_bars(
            aurocs=syn_aurocs, 
            errs=syn_sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_SYN__syn_n_samples_{n_syn_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions on synthetic data",
            colors=colors,
            hatches=['|'] * len(syn_aurocs)
        )
        plot_aurocs_with_error_bars(
            aurocs=syn_aurocs, 
            errs=syn_sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_SYN_with_annotations__syn_n_samples_{n_syn_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions on synthetic data",
            colors=colors,
            hatches=['|'] * len(syn_aurocs),
            add_annotations=True
        )



def evaluate_vae_conditioning_experiment(
    model_dir: Path, 
    latent_dim: int,
    dec_hidden_dim: int,
    pattern_seeds: list,
    plot_dir: Path,
    n_seeds,
    n_syn_samples: int, 
    bootstrap_iters: int = 1_000,
    metric: str = "auroc",
) -> None:

    plot_dir.mkdir(parents=True, exist_ok=True)

    aurocs = []
    sems = []
    syn_aurocs = []
    syn_sems = []
    x_labels = []

    all_seed_aurocs = []

    name2dir = {
        "Separate \nModels": f"vae__conditioning_None__phoneme_cls_3__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",

        "FiLM": f"vae__conditioning_film__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_None__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",

        # "Concat 2": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_2__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        # "Concat 4": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_4__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 8": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_8__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 16": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_16__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 32": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_32__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 64": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_64__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
        "Concat 128": f"vae__conditioning_concat__phoneme_cls_3_31__latent_dim_{latent_dim}__dec_emb_dim_128__dec_hidden_dim_{dec_hidden_dim}__seed_{{pattern_seed}}",
    }
    for model_name, weight_dir in name2dir.items():
        seed_aurocs = []
        syn_seed_aurocs = []

        for pattern_seed in pattern_seeds:
            sub_dir = model_dir / weight_dir.format(pattern_seed=pattern_seed) / "phoneme_classifier"
            print(f"sub_dir = {sub_dir}")

            for out_dir in sub_dir.iterdir():
                if out_dir.is_dir():
                    seed_number = re.search(r'seed_(\d+)', str(out_dir).split("/")[-1])
                    seed_value = int(seed_number.group(1)) if seed_number else None
                    if f"syn_n_samples_{n_syn_samples}" in str(out_dir) and seed_value in range(n_seeds):
                        print(f"out_dir = {out_dir}")

                        with open(out_dir / "ouput.json", 'r') as f:
                            output = json.load(f)
                        
                        auroc, _ = compute_auroc_with_stderr(
                            y_true=output["y_true"], 
                            y_pred=output["y_pred"], 
                            n_iters=bootstrap_iters, 
                        )
                        seed_aurocs.append(auroc)

                        syn_auroc, _ = compute_auroc_with_stderr(
                            y_true=output["y_true_syn"], 
                            y_pred=output["y_pred_syn"], 
                            n_iters=bootstrap_iters, 
                        )
                        syn_seed_aurocs.append(syn_auroc)

        all_seed_aurocs.append(seed_aurocs)

        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=len(seed_aurocs), replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        auroc = np.mean(seed_aurocs)
        sem = np.std(bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        aurocs.append(auroc)
        sems.append(sem)

        syn_bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(syn_seed_aurocs, size=len(seed_aurocs), replace=True)
            syn_bootstrapped_aurocs.append(np.mean(sample))

        syn_auroc = np.mean(syn_seed_aurocs)
        syn_sem = np.std(syn_bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        syn_aurocs.append(syn_auroc)
        syn_sems.append(syn_sem)

        x_labels.append(model_name)


    colors = sns.color_palette("Set2", 3)[2]

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    plot_aurocs_with_error_bars(
        aurocs=aurocs, 
        errs=sems, 
        x_labels=x_labels, 
        out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_REAL__syn_n_samples_{n_syn_samples}.png",
        xlabel="Conditiong mechanism",
        title="AUROC performance of VAEs with varying conditioning mechanisms \non real data",
        colors=colors,
        figsize=(11, 6),
        rotate_xlabels=True
    )
    plot_aurocs_with_error_bars(
        aurocs=aurocs, 
        errs=sems, 
        x_labels=x_labels, 
        out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_REAL_with_annotations__syn_n_samples_{n_syn_samples}.png",
        xlabel="Conditiong mechanism",
        title="AUROC performance of VAEs with varying conditioning mechanisms \non real data",
        colors=colors,
        add_annotations=True,
        figsize=(11, 6),
        rotate_xlabels=True
    )
    model_auroc_dict = dict(zip(x_labels, all_seed_aurocs))
    output_file = plot_dir / f"{timestamp}__SEM_aurocs_with_err__seed_{n_seeds}.json"
    with open(output_file, "w") as f:
        json.dump(model_auroc_dict, f, indent=4)


    plot_aurocs_with_error_bars(
        aurocs=syn_aurocs, 
        errs=syn_sems, 
        x_labels=x_labels, 
        out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_SYN__syn_n_samples_{n_syn_samples}.png",
        xlabel="Conditiong mechanism",
        title="AUROC performance of VAEs with varying conditioning mechanisms \non synthetic data",
        colors=colors,
        hatches=['|'] * len(syn_aurocs),
        figsize=(11, 6),
        rotate_xlabels=True
    )
    plot_aurocs_with_error_bars(
        aurocs=syn_aurocs, 
        errs=syn_sems, 
        x_labels=x_labels, 
        out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_SYN_with_annotations__syn_n_samples_{n_syn_samples}.png",
        xlabel="Conditiong mechanism",
        title="AUROC performance of VAEs with varying conditioning mechanisms \non synthetic data",
        colors=colors,
        hatches=['|'] * len(syn_aurocs),
        add_annotations=True,
        figsize=(11, 6),
        rotate_xlabels=True
    )




def eval_vae_reconstruction_cap(weights_path):
    # load a VAE model from this weightsdir
    model = load_t2b_gen_model(weights_path=weights_path)
    args = {}
    data_dir = Path("/data/engs-pnpl/lina4471/willett2023/competitionData")
    args["train_set_path"] = str(data_dir / "rnn_train_set_with_logits.pkl")

    args["phoneme_cls"] = [3, 31]
    args["correctness_value"] = ["C"]
    args["seed"] = 0
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = args["device"]
    set_seeds(args["seed"])

    #load the real train data
    phoneme_ds_filter = {"correctness_value": args["correctness_value"], "phoneme_cls": args["phoneme_cls"]}

    transform = None
    args["transform"] = "softsign"
    if args["transform"] == "softsign":
        transform = transforms.Compose(
            [
                TransposeTransform(0, 1),
                ReorderChannelTransform(),
                AddOneDimensionTransform(dim=0),
                GaussianSmoothing(
                    256,
                    kernel_size=20.0,
                    sigma=2.0,
                    dim=1,
                ),
                SoftsignTransform(),
            ]
        )
    

    # load real val set
    train_dl_real = get_data_loader(
        data=load_pkl(args["train_set_path"]),
        batch_size=1,
        shuffle=True,
        collate_fn=None,
        dataset_cls=PhonemeDataset,
        phoneme_ds_filter=phoneme_ds_filter,
        class_weights=None,
        transform=transform,
    )
    train_dl_real.name = "train-real"
    train_ds = train_dl_real.dataset


    originals = []
    reconstructions = []
    originals_ah = []
    originals_t = []
    reconstructions_ah = []
    reconstructions_t = []

    for j, batch in enumerate(train_dl_real):
        X, y, _, _ = batch
        X = X.to(device)
        y = y.to(device)
        recon, _, _ = model(X, y)

        recon = recon.cpu().detach().numpy().reshape(X.shape[0], -1)
        original = X.cpu().detach().numpy().reshape(X.shape[0], -1)
        reconstructions.append(recon)
        originals.append(original)

        n_samples = 10_000
        if y.item() == 3:
            if len(originals_ah) < n_samples:
                originals_ah.append(original)
                reconstructions_ah.append(recon)
        else:
            if len(originals_t) < n_samples:
                originals_t.append(original)
                reconstructions_t.append(recon)

        if j % 500 == 0:
            print(f"j = {j}")
            print(f"len(originals_ah) = {len(originals_ah)}")
            print(f"len(originals_t) = {len(originals_t)}")

        if len(originals_t) == n_samples and len(originals_ah) == n_samples:
            break


    originals_ah = np.vstack(originals_ah)
    reconstructions_ah = np.vstack(reconstructions_ah)

    data_ah = np.vstack([reconstructions_ah, originals_ah])

    correlation_matrix_ah = np.corrcoef(data_ah)

    n_samples_ah = len(reconstructions_ah)
    reconstruction_vs_original_ah = correlation_matrix_ah[:n_samples_ah, n_samples_ah:]

    diagonal_correlations_ah = np.diag(reconstruction_vs_original_ah)
    off_diagonal_correlations_ah = reconstruction_vs_original_ah[np.triu_indices_from(reconstruction_vs_original_ah, k=1)]


    print(f"AH Average Diagonal Correlation: {np.mean(diagonal_correlations_ah)}")
    print(f"AH Average Off-Diagonal Correlation: {np.mean(off_diagonal_correlations_ah)}")


    originals_t = np.vstack(originals_t)
    reconstructions_t = np.vstack(reconstructions_t)

    data_t = np.vstack([reconstructions_t, originals_t])

    correlation_matrix_t = np.corrcoef(data_t)

    n_samples_t = len(reconstructions_t)
    reconstruction_vs_original_t = correlation_matrix_t[:n_samples_t, n_samples_t:]

    diagonal_correlations_t = np.diag(reconstruction_vs_original_t)
    off_diagonal_correlations_t = reconstruction_vs_original_t[np.triu_indices_from(reconstruction_vs_original_t, k=1)]

    print(f"T Average Diagonal Correlation: {np.mean(diagonal_correlations_t)}")
    print(f"T Average Off-Diagonal Correlation: {np.mean(off_diagonal_correlations_t)}")

    plt.figure(figsize=(14, 8))
    # plt.colorbar()

    # Plot for _ah
    plt.subplot(1, 2, 1)
    plt.imshow(reconstruction_vs_original_ah, cmap="coolwarm", interpolation='nearest')
    plt.title('Phoneme AH', fontsize=18)
    plt.xlabel('Original Images', fontsize=16)
    plt.ylabel('Reconstructed Images', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.gca().set_aspect('auto')

    # Plot for _t
    plt.subplot(1, 2, 2)
    plt.imshow(reconstruction_vs_original_t, cmap="coolwarm", interpolation='nearest')
    plt.title('Phoneme T', fontsize=18)
    plt.xlabel('Original Images', fontsize=16)
    plt.ylabel('Reconstructed Images', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.gca().set_aspect('auto')

    # Show the plots
    plt.tight_layout()
    # plt.colorbar()
    plt.suptitle('Correlation Matrix: VAE Reconstructions vs Originals', fontsize=18, fontweight='bold')
    plt.savefig(Path(f"/data/engs-pnpl/lina4471/repos/neural_seq_decoder/evaluation/correlation_plots/corr_vae_reconstructions_n_samples_{j}_both_classes.png"))

    



def evaluate_final_experiment(
    vae_dir: Path, 
    vae_pattern: str,
    gan_dir: Path, 
    gan_pattern: str, 
    pattern_seeds: list, 
    n_gen_samples: int, 
    n_seeds: int = 10, 
    bootstrap_iters: int = 10,
    prefix=""
):

    aurocs = []
    sems = []
    syn_aurocs = []
    syn_sems = []

    all_seed_aurocs = []
    all_test_trues = []

    x_labels = []
    
    model_names = ["Real data", "GAN data", "VAE data"]
    thresholds = [0.4, 0.5, 0.6,]
    label2miscls = {t: {m: [] for m in model_names} for t in thresholds}

    for pattern_seed in pattern_seeds:
        sub_dir = model_dir / file_name_pattern.format(latent_dim=latent_dim, seed=pattern_seed) / "phoneme_classifier"
        

    for model_dir, model_name in zip([None, gan_dir, vae_dir], model_names):
        seed_aurocs = []
        syn_seed_aurocs = []
        model_seeds = [None] if model_dir is None else pattern_seeds

        for i, p_seed in enumerate(model_seeds):
            if model_dir is not None:
                if "GAN" in model_name:
                    sub_dir = model_dir / gan_pattern.format(seed=p_seed) / "phoneme_classifier"
                elif "VAE" in model_name:
                    sub_dir = model_dir / vae_pattern.format(seed=p_seed) / "phoneme_classifier"

                out_dir = sub_dir / "phoneme_classifier"

            elif model_dir is None:
                out_dir = Path("/data/engs-pnpl/lina4471/willett2023/generative_models/experiments") / "classifier_trained_on_real" / "phoneme_classifier"
                
            for out_dir in sub_dir.iterdir():
                if out_dir.is_dir():
                    if f"syn_n_samples_{n_syn_samples}" in str(out_dir):

                        with open(out_dir / "ouput.json", 'r') as f:
                            output = json.load(f)
                        
                        auroc, _ = compute_auroc_with_stderr(
                            y_true=output["y_true"], 
                            y_pred=output["y_pred"], 
                            n_iters=bootstrap_iters, 
                        )
                        seed_aurocs.append(auroc)

                        if model_name != "Real data":
                            syn_auroc, _ = compute_auroc_with_stderr(
                                y_true=output["y_true_syn"], 
                                y_pred=output["y_pred_syn"], 
                                n_iters=bootstrap_iters, 
                            )
                            syn_seed_aurocs.append(syn_auroc)

                        for t in thresholds:
                            miscls_indices = get_miclassified_indices(
                                y_true=output["test_y_true"],  
                                y_pred=output["test_y_pred"],
                                threshold=t,
                            )
                            label2miscls[t][model_name] = miscls_indices
                            all_test_trues.append(output["test_y_true"])

        bootstrapped_aurocs = []
        for _ in range(bootstrap_iters):
            sample = np.random.choice(seed_aurocs, size=len(seed_aurocs), replace=True)
            bootstrapped_aurocs.append(np.mean(sample))

        auroc = np.mean(seed_aurocs)
        sem = np.std(bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
        aurocs.append(auroc)
        sems.append(sem)

        if model_name != "Real data":
            syn_bootstrapped_aurocs = []
            for _ in range(bootstrap_iters):
                sample = np.random.choice(syn_seed_aurocs, size=len(seed_aurocs), replace=True)
                syn_bootstrapped_aurocs.append(np.mean(sample))

            syn_auroc = np.mean(syn_seed_aurocs)
            syn_sem = np.std(syn_bootstrapped_aurocs) / np.sqrt(bootstrap_iters)
            syn_aurocs.append(syn_auroc)
            syn_sems.append(syn_sem)

        x_labels.append(model_name)

    colors = sns.color_palette("Set2", len(x_labels))

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    plot_dir = ROOT_DIR / "evaluation" / "experiments" / "final_experiment" / "plots"

    for i, figsize in enumerate([(8, 6), (7, 6), (6, 6), (6, 5)]):

        plot_aurocs_with_error_bars(
            aurocs=aurocs, 
            errs=sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_{i}__{prefix}__SEM_aurocs_with_err__seed_{n_seeds}__n_gen_samples_{n_gen_samples}.png",
            xlabel="Training Data",
            title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
            colors=colors,
            figsize=figsize,
        )
        
        plot_aurocs_with_error_bars(
            aurocs= aurocs, 
            errs=sems, 
            x_labels=x_labels, 
            out_file=plot_dir / f"{timestamp}_{i}__{prefix}__SEM_aurocs_with_err__seed_{n_seeds}__n_gen_samples_{n_gen_samples}_with_annotations.png",
            xlabel="Training Data",
            title="AUROC of Phoneme Classifier trained on Real vs. Synthetic Data",
            colors=colors,
            figsize=figsize,
            add_annotations=True
        )

        plot_aurocs_with_error_bars(
            aurocs=syn_aurocs, 
            errs=syn_sems, 
            x_labels=x_labels[1:], 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_SYN__syn_n_samples_{n_syn_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions on synthetic data",
            colors=colors,
            hatches=['|'] * len(syn_aurocs)
        )
        plot_aurocs_with_error_bars(
            aurocs=syn_aurocs, 
            errs=syn_sems, 
            x_labels=x_labels[:1], 
            out_file=plot_dir / f"{timestamp}_SEM_aurocs_with_err__seed_{n_seeds}_SYN_with_annotations__syn_n_samples_{n_syn_samples}.png",
            xlabel="latent dim",
            title="AUROC performance of VAEs with varying latent dimensions on synthetic data",
            colors=colors,
            hatches=['|'] * len(syn_aurocs),
            add_annotations=True
        )
    
    model_auroc_dict = dict(zip(x_labels, all_seed_aurocs))
    output_file = plot_dir / f"{timestamp}__{prefix}__SEM_aurocs_with_err__seed_{n_seeds}__n_gen_samples_{n_gen_samples}.json"
    with open(output_file, "w") as f:
        json.dump(model_auroc_dict, f, indent=4)

    for t in thresholds:
        plot_venn3(label2miscls[t], plot_dir / "venn_plots" / f"venn3_{timestamp}__{prefix}__threshold_{t}.png", title=f"Overlap of Misclassified Samples Across Models \n(threshold = {t})")

    print(f"np.array_equal(all_test_trues[0], all_test_trues[1]) = {np.array_equal(all_test_trues[0], all_test_trues[1])}")
    print(f"np.array_equal(all_test_trues[0], all_test_trues[2]) = {np.array_equal(all_test_trues[0], all_test_trues[2])}")




if __name__ == "__main__":
    print("in main ...")

    vae_latent_dim_experiment = False
    vae_conditioning_experiment = False

    vae_reconstruction_capabilities = True

    gan_mode_collapse = False

    if vae_latent_dim_experiment:
        for n_seeds in [10,]:
            for dhd in [256, 512]:
                for n_syn_samples in [10_000, 30_000, 50_000]:
                    evaluate_vae_latent_dim_experiment(
                        model_dir=Path(f"/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_latent_dim_cond_bn_True__dhd_{dhd}"), 
                        file_name_pattern="vae__latent_dim_{latent_dim}__seed_{seed}", 
                        pattern_latent_dims=[32, 64, 128, 256, 512, 1024], 
                        pattern_seeds=list(range(n_seeds)),
                        n_syn_samples = n_syn_samples,
                        plot_dir= ROOT_DIR / "evaluation" / "experiments" / f"vae_latent_dim_cond_bn_True__dhd_{dhd}" / "plots"
                    )

    if vae_conditioning_experiment:
        for n_seeds in [10,]:
            for latent_dim in [256, 512]:
                for dec_hidden_dim in [256, 512,]:
                    for n_syn_samples in [10_000]:
                        model_dir = Path(f"/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_conditioning_cond_bn_True/ld_{latent_dim}_dhd_{dec_hidden_dim}")
                        evaluate_vae_conditioning_experiment(
                            model_dir=model_dir, 
                            latent_dim=latent_dim,
                            dec_hidden_dim=dec_hidden_dim,
                            pattern_seeds=list(range(10)),
                            n_seeds=n_seeds,
                            n_syn_samples=n_syn_samples,
                            plot_dir=ROOT_DIR / "evaluation" / "experiments" / "vae_conditioning_cond_bn_True" / f"ld_{latent_dim}_dhd_{dec_hidden_dim}" / "plots"
                        )
        
    if vae_reconstruction_capabilities:
        eval_vae_reconstruction_cap(weights_path=Path("/data/engs-pnpl/lina4471/willett2023/generative_models/experiments/vae_conditioning_cond_bn_True/ld_256_dhd_256/vae__conditioning_film__phoneme_cls_3_31__latent_dim_256__dec_emb_dim_None__dec_hidden_dim_256__seed_0/modelWeights_epoch_82"))

    if gan_mode_collapse:
        pass

    if final_experiment:
        evaluate_final_experiment()
