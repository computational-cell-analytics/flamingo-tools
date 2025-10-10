import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import SYNAPSE_DIR_ROOT

FILE_EXTENSION = "png"
png_dpi = 300


def _load_ribbon_synapse_counts():
    # TODO update the version!
    ihc_version = "ihc_counts_v4b"
    table_path = os.path.join(SYNAPSE_DIR_ROOT, ihc_version, "ihc_count_M_AMD_OTOF1_L.tsv")
    x = pd.read_csv(table_path, sep="\t")
    syn_counts = x.synapse_count.values.tolist()
    return syn_counts


def fig_05c(save_path, plot=False):
    """Bar plot showing the IHC count and distribution of synapse markers per IHC segmentation over OTOF cochlea.
    """
    # TODO update the alias.
    # For MOTOF1L
    alias = "M10L"

    main_label_size = 20
    main_tick_size = 12
    htext_size = 10

    ribbon_synapse_counts = _load_ribbon_synapse_counts()

    rows, columns = 1, 2
    fig, axes = plt.subplots(rows, columns, figsize=(columns*4, rows*4))

    #
    # Create the plot for IHCs.
    #
    ihc_values = [len(ribbon_synapse_counts)]

    ylim0 = 600
    ylim1 = 800
    y_ticks = [i for i in range(600, 800 + 1, 100)]

    axes[0].set_ylabel("IHC count", fontsize=main_label_size)
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    axes[0].set_ylim(ylim0, ylim1)

    axes[0].boxplot(ihc_values)
    axes[0].set_xticklabels([alias], fontsize=main_label_size)

    # Set the reference values for healthy cochleae
    xmin = 0.5
    xmax = 1.5
    ihc_reference_values = [712, 710, 721, 675]  # MLR226L, MLR226R, MLR227L, MLR227R

    ihc_value = np.mean(ihc_reference_values)
    ihc_std = np.std(ihc_reference_values)

    upper_y = ihc_value + 1.96 * ihc_std
    lower_y = ihc_value - 1.96 * ihc_std

    axes[0].hlines([lower_y, upper_y], xmin, xmax, colors=["C1" for _ in range(2)])
    axes[0].text(1, upper_y + 10, "healthy cochleae", color="C1", fontsize=main_tick_size, ha="center")
    axes[0].fill_between([xmin, xmax], lower_y, upper_y, color="C1", alpha=0.05, interpolate=True)

    #
    # Create the plot for ribbon synapse distribution.
    #
    ylim0 = -1
    ylim1 = 24
    y_ticks = [i for i in range(0, 25, 5)]

    axes[1].set_ylabel("Ribbon Syn. per IHC", fontsize=main_label_size)
    axes[1].set_yticks(y_ticks)
    axes[1].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    axes[1].set_ylim(ylim0, ylim1)

    axes[1].boxplot(ribbon_synapse_counts)
    axes[1].set_xticklabels([alias], fontsize=main_label_size)

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_ticks_position("right")
    axes[1].yaxis.set_label_position("right")

    # Set the reference values for healthy cochleae
    xmin = 0.5
    xmax = 1.5
    syn_reference_values = [14.1, 12.7, 13.8, 13.4]  # MLR226L, MLR226R, MLR227L, MLR227R

    syn_value = np.mean(syn_reference_values)
    syn_std = np.std(syn_reference_values)

    upper_y = syn_value + 1.96 * syn_std
    lower_y = syn_value - 1.96 * syn_std

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=["C1" for _ in range(2)])
    plt.text(
        1.25, upper_y + 0.01*axes[1].get_ylim()[1]-axes[1].get_ylim()[0], "healthy cochleae",
        color="C1", fontsize=htext_size, ha="center"
    )
    plt.fill_between([xmin, xmax], lower_y, upper_y, color="C1", alpha=0.05, interpolate=True)

    # Save and plot the figure.
    plt.tight_layout()
    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


# TODO
def fig_05d(save_path, plot):
    if False:
        s3 = create_s3_target()

        # Intensity distribution for OTOF
        cochlea = "M_AMD_OTOF1_L"
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the seg table and filter the compartments.
        source_name = "IHC_v4c"
        source = sources[source_name]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")
        print(table)

    # TODO would need the new intensity subtracted data here.
    # Reference: intensity distributions for ChReef


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 5 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig5")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: Monitoring of the Syn / IHC loss
    # fig_05c(save_path=os.path.join(args.figure_dir, "fig_05c"), plot=args.plot)

    # Panel D: Tonotopic mapping of the intensities.
    fig_05d(save_path=os.path.join(args.figure_dir, f"fig_05d.{FILE_EXTENSION}"), plot=args.plot)


if __name__ == "__main__":
    main()
