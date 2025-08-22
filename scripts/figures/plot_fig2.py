import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from matplotlib import colors
from skimage.segmentation import find_boundaries

from util import literature_reference_values

png_dpi = 300


def scramble_instance_labels(arr):
    """Scramble indexes of instance segmentation to avoid neighboring colors.
    """
    unique = list(np.unique(arr)[1:])
    rng = np.random.default_rng(seed=42)
    new_list = rng.uniform(1, len(unique) + 1, size=(len(unique)))
    new_arr = np.zeros(arr.shape)
    for old_id, new_id in zip(unique, new_list):
        new_arr[arr == old_id] = new_id
    return new_arr


def plot_seg_crop(img_path, seg_path, save_path, xlim1, xlim2, ylim1, ylim2, boundary_rgba=[0, 0, 0, 0.5], plot=False):
    seg = tifffile.imread(seg_path)
    if len(seg.shape) == 3:
        seg = seg[10, xlim1:xlim2, ylim1:ylim2]
    else:
        seg = seg[xlim1:xlim2, ylim1:ylim2]

    img = tifffile.imread(img_path)
    img = img[10, xlim1:xlim2, ylim1:ylim2]

    # create color map with random distribution for coloring instance segmentation
    unique = list(np.unique(seg)[1:])
    n_instances = len(unique)

    seg = scramble_instance_labels(seg)

    rng = np.random.default_rng(seed=42)   # fixed seed for reproducibility
    colors_array = rng.uniform(0, 1, size=(n_instances, 4))  # RGBA values in [0,1]
    colors_array[:, 3] = 1.0  # full alpha
    colors_array[0, 3] = 0.0  # make label 0 transparent (background)
    cmap = colors.ListedColormap(colors_array)

    boundaries = find_boundaries(seg, mode="inner")
    boundary_overlay = np.zeros((*boundaries.shape, 4))

    boundary_overlay[boundaries] = boundary_rgba  # RGBA = black

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.imshow(seg, cmap=cmap, alpha=0.5, interpolation="nearest")
    ax.imshow(boundary_overlay)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_02b_sgn(save_dir, plot=False):
    """Plot crops of SGN segmentation of CochleaNet, Cellpose and micro-sam.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    val_sgn_dir = f"{cochlea_dir}/predictions/val_sgn"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation"

    crop_name = "MLR169R_PV_z3420_allturns_full"
    img_path = os.path.join(image_dir, f"{crop_name}.tif")

    xlim1 = 2000
    xlim2 = 2500
    ylim1 = 3100
    ylim2 = 3600
    boundary_rgba = [1, 1, 1, 0.5]

    for seg_net in ["distance_unet", "cellpose-sam", "micro-sam"]:
        save_path = os.path.join(save_dir, f"fig_02b_sgn_{seg_net}.png")
        seg_dir = os.path.join(val_sgn_dir, seg_net)
        seg_path = os.path.join(seg_dir, f"{crop_name}_seg.tif")

        plot_seg_crop(img_path, seg_path, save_path, xlim1, xlim2, ylim1, ylim2, boundary_rgba, plot=plot)


def fig_02b_ihc(save_dir, plot=False):
    """Plot crops of IHC segmentation of CochleaNet, Cellpose and micro-sam.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    val_sgn_dir = f"{cochlea_dir}/predictions/val_ihc"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationIHCs"

    crop_name = "MLR226L_VGlut3_z1200_3turns_full"
    img_path = os.path.join(image_dir, f"{crop_name}.tif")

    xlim1 = 1900
    xlim2 = 2400
    ylim1 = 2000
    ylim2 = 2500
    boundary_rgba = [1, 1, 1, 0.5]

    for seg_net in ["distance_unet_v4b", "cellpose-sam", "micro-sam"]:
        save_path = os.path.join(save_dir, f"fig_02b_ihc_{seg_net}.png")
        seg_dir = os.path.join(val_sgn_dir, seg_net)
        seg_path = os.path.join(seg_dir, f"{crop_name}_seg.tif")

        plot_seg_crop(img_path, seg_path, save_path, xlim1, xlim2, ylim1, ylim2, boundary_rgba, plot=plot)


def fig_02c(save_path, plot=False, all_versions=False):
    """Scatter plot showing the precision, recall, and F1-score of SGN (distance U-Net, manual),
    IHC (distance U-Net, manual), and synapse detection (U-Net).
    """
    # precision, recall, f1-score
    sgn_unet = [0.887, 0.88, 0.884]
    sgn_annotator = [0.95, 0.849, 0.9]

    ihc_v4b = [0.91, 0.819, 0.862]
    ihc_v4c = [0.905, 0.831, 0.866]
    ihc_v4c_filter = [0.919, 0.775, 0.841]

    ihc_annotator = [0.958, 0.956, 0.957]
    syn_unet = [0.931, 0.905, 0.918]

    # This is the version with IHC v4b segmentation:
    # 4th version of the network with optimized segmentation params
    version_1 = [sgn_unet, sgn_annotator, ihc_v4b, ihc_annotator, syn_unet]
    settings_1 = ["automatic", "manual", "automatic", "manual", "automatic"]

    # This is the version with IHC v4c segmentation:
    # 4th version of the network with optimized segmentation params and split of falsely merged IHCs
    version_2 = [sgn_unet, sgn_annotator, ihc_v4c, ihc_annotator, syn_unet]
    settings_2 = ["automatic", "manual", "automatic", "manual", "automatic"]

    # This is the version with IHC v4c + filter segmentation:
    # 4th version of the network with optimized segmentation params and split of falsely merged IHCs
    # + filtering out IHCs with zero mapped synapses.
    version_3 = [sgn_unet, sgn_annotator, ihc_v4c_filter, ihc_annotator, syn_unet]
    settings_3 = ["automatic", "manual", "automatic", "manual", "automatic"]

    if all_versions:
        versions = [version_1, version_2, version_3]
        settings = [settings_1, settings_2, settings_3]
        save_suffix = ["_v4b", "_v4c", "_v4c_filter"]
        save_paths = [save_path + i for i in save_suffix]
    else:
        versions = [version_2]
        settings = [settings_2]
        save_suffix = ["_v4c"]
        save_paths = [save_path + i for i in save_suffix]

    for version, setting, save_path in zip(versions, settings, save_paths):
        precision = [i[0] for i in version]
        recall = [i[1] for i in version]
        f1score = [i[2] for i in version]

        descr_y = 0.72

        # Convert setting labels to numerical x positions
        x = np.array([0.8, 1.2, 1.8, 2.2, 3])
        offset = 0.08  # horizontal shift for scatter separation

        # Plot
        plt.figure(figsize=(8, 5))

        main_label_size = 20
        sub_label_size = 16
        main_tick_size = 12
        legendsize = 16

        plt.scatter(x - offset, precision, label="Precision", marker="o", s=80)
        plt.scatter(x,         recall, label="Recall", marker="^", s=80)
        plt.scatter(x + offset, f1score, label="F1-score", marker="*", s=80)

        plt.text(1, descr_y, "SGN", fontsize=main_label_size, horizontalalignment="center")
        plt.text(2, descr_y, "IHC", fontsize=main_label_size, horizontalalignment="center")
        plt.text(3, descr_y, "Synapse", fontsize=main_label_size, horizontalalignment="center")

        # Labels and formatting
        plt.xticks(x, setting, fontsize=sub_label_size)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Value", fontsize=main_label_size)
        plt.ylim(0.76, 1)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11),
                   ncol=3, fancybox=True, shadow=False, framealpha=0.8, fontsize=legendsize)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)

        if plot:
            plt.show()
        else:
            plt.close()


# Load the synapse counts for all IHCs from the relevant tables.
def _load_ribbon_synapse_counts():
    ihc_version = "ihc_counts_v4c"
    synapse_dir = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/{ihc_version}"
    tables = [entry.path for entry in os.scandir(synapse_dir) if "ihc_count_M_LR" in entry.name]
    syn_counts = []
    for tab in tables:
        x = pd.read_csv(tab, sep="\t")
        syn_counts.extend(x["synapse_count"].values.tolist())
    return syn_counts


def fig_02d_01(save_path, plot=False, all_versions=False, plot_average_ribbon_synapses=False):
    """Box plot showing the counts for SGN and IHC per (mouse) cochlea in comparison to literature values.
    """
    main_tick_size = 16
    main_label_size = 24

    rows = 1
    columns = 3 if plot_average_ribbon_synapses else 2

    sgn_values = [11153, 11398, 10333, 11820]
    ihc_v4b_values = [836, 808, 796, 901]
    ihc_v4c_values = [712, 710, 721, 675]
    ihc_v4c_filtered_values = [562, 647, 626, 628]

    if all_versions:
        ihc_list = [ihc_v4b_values, ihc_v4c_values, ihc_v4c_filtered_values]
        suffixes = ["_v4b", "_v4c", "_v4c_filtered"]
        assert not plot_average_ribbon_synapses
    else:
        ihc_list = [ihc_v4c_values]
        suffixes = ["_v4c"]

    for (ihc_values, suffix) in zip(ihc_list, suffixes):
        fig, axes = plt.subplots(rows, columns, figsize=(columns*4, rows*4))
        ax = axes.flatten()

        save_path_new = save_path + suffix
        ax[0].boxplot(sgn_values)
        ax[1].boxplot(ihc_values)

        # Labels and formatting
        ax[0].set_xticklabels(["SGN"], fontsize=main_label_size)

        ylim0 = 8500
        ylim1 = 12500
        y_ticks = [i for i in range(9000, 12000 + 1, 1000)]

        ax[0].set_ylabel("Count per cochlea", fontsize=main_label_size)
        ax[0].set_yticks(y_ticks)
        ax[0].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
        ax[0].set_ylim(ylim0, ylim1)
        ax[0].yaxis.set_ticks_position("left")

        # set range of literature values
        xmin = 0.5
        xmax = 1.5
        ax[0].set_xlim(xmin, xmax)
        lower_y, upper_y = literature_reference_values("SGN")
        ax[0].hlines([lower_y, upper_y], xmin, xmax)
        ax[0].text(1.1, (lower_y + upper_y) // 2, "literature", color="C0", fontsize=main_tick_size, ha="left")
        ax[0].fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

        ylim0 = 600
        ylim1 = 800
        y_ticks = [i for i in range(600, 800 + 1, 100)]

        ax[1].set_xticklabels(["IHC"], fontsize=main_label_size)
        ax[1].set_yticks(y_ticks)
        ax[1].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
        ax[1].set_ylim(ylim0, ylim1)
        if not plot_average_ribbon_synapses:
            ax[1].yaxis.tick_right()
            ax[1].yaxis.set_ticks_position("right")

        # set range of literature values
        xmin = 0.5
        xmax = 1.5
        lower_y, upper_y = literature_reference_values("IHC")
        ax[1].set_xlim(xmin, xmax)
        ax[1].hlines([lower_y, upper_y], xmin, xmax)
        ax[1].text(1.1, (lower_y + upper_y) // 2, "literature", color="C0", fontsize=main_tick_size, ha="left")
        ax[1].fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

        if plot_average_ribbon_synapses:
            ribbon_synapse_counts = _load_ribbon_synapse_counts()
            ylim0 = -1
            ylim1 = 41
            y_ticks = [0, 10, 20, 30, 40, 50]

            ax[2].boxplot(ribbon_synapse_counts)
            ax[2].set_xticklabels(["Ribbon Syn. per IHC"], fontsize=main_label_size)
            ax[2].set_yticks(y_ticks)
            ax[2].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
            ax[2].set_ylim(ylim0, ylim1)

            # set range of literature values
            xmin = 0.5
            xmax = 1.5
            lower_y, upper_y = literature_reference_values("synapse")
            ax[2].set_xlim(xmin, xmax)
            ax[2].hlines([lower_y, upper_y], xmin, xmax)
            ax[2].text(1.1, (lower_y + upper_y) // 2, "literature", color="C0", fontsize=main_tick_size, ha="left")
            ax[2].fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

        plt.tight_layout()
        plt.savefig(save_path_new, dpi=png_dpi)

        if plot:
            plt.show()
        else:
            plt.close()


def fig_02d_02(save_path, filter_zeros=True, plot=False):
    """Bar plot showing the distribution of synapse markers per IHC segmentation average over multiple clochleae.
    """
    cochleae = ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"]
    ihc_version = "ihc_counts_v4b"
    synapse_dir = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/{ihc_version}"

    max_dist = 3
    bins = 10
    cap = 30
    plot_density = False

    results = []
    for cochlea in cochleae:
        synapse_file = os.path.join(synapse_dir, f"ihc_count_{cochlea}.tsv")
        ihc_table = pd.read_csv(synapse_file, sep="\t")
        syn_per_ihc = list(ihc_table["synapse_count"])
        if filter_zeros:
            syn_per_ihc = [s for s in syn_per_ihc if s != 0]
        results.append(syn_per_ihc)

    results = [np.clip(r, 0, cap) for r in results]

    # Define bins (shared for all datasets)
    bins = np.linspace(0, 30, 11)  # 29 bins

    # Compute histogram (relative) for each dataset
    histograms = []
    for data in results:
        counts, _ = np.histogram(data, bins=bins, density=plot_density)
        histograms.append(counts)

    histograms = np.array(histograms)

    # Compute mean and std for each bin across datasets
    mean_counts = histograms.mean(axis=0)
    std_counts = histograms.std(axis=0)

    # Get bin centers for plotting
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, mean_counts, width=(bins[1] - bins[0]), yerr=std_counts, alpha=0.7, capsize=4,
            label="Mean ± Std Dev", edgecolor="black")

    main_label_size = 20
    main_tick_size = 16
    legendsize = 16

    # Labels and formatting
    x_ticks = [i for i in range(0, cap + 1, 5)]
    if plot_density:
        y_ticks = [i * 0.02 for i in range(0, 10, 2)]
    else:
        y_ticks = [i for i in range(0, 300, 50)]

    plt.xticks(x_ticks, fontsize=main_tick_size)
    plt.yticks(y_ticks, fontsize=main_tick_size)
    if plot_density:
        plt.ylabel("Proportion of IHCs [%]", fontsize=main_label_size)
    else:
        plt.ylabel("Number of IHCs", fontsize=main_label_size)
    plt.xlabel(f"Ribbon Synapses per IHC @ {max_dist} µm", fontsize=main_label_size)

    plt.title("Average Synapses per IHC for a Dataset of 4 Cochleae")

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=legendsize)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)

    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 2 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig2")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: Evaluation of the segmentation results:
    fig_02b_sgn(save_dir=args.figure_dir, plot=args.plot)
    fig_02b_ihc(save_dir=args.figure_dir, plot=args.plot)
    fig_02c(save_path=os.path.join(args.figure_dir, "fig_02c"), plot=args.plot, all_versions=False)

    # Panel D: The number of SGNs, IHCs and average number of ribbon synapses per IHC
    fig_02d_01(save_path=os.path.join(args.figure_dir, "fig_02d"), plot=args.plot, plot_average_ribbon_synapses=True)

    # Alternative version of synapse distribution for panel D.
    # fig_02d_02(save_path=os.path.join(args.figure_dir, "fig_02d_02"), plot=args.plot)
    # fig_02d_02(save_path=os.path.join(args.figure_dir, "fig_02d_02_v4c"), filter_zeros=False, plot=plot)
    # fig_02d_02(save_path=os.path.join(args.figure_dir, "fig_02d_02_v4c_filtered"), filter_zeros=True, plot=plot)
    # fig_02d_02(save_path=os.path.join(args.figure_dir, "fig_02d_02_v4b"), filter_zeros=True, plot=args.plot)


if __name__ == "__main__":
    main()
