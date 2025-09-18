import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from util import literature_reference_values_gerbil, SYNAPSE_DIR_ROOT

FILE_EXTENSION = "png"
png_dpi = 300


# Load the synapse counts for all IHCs from the relevant tables.
def _load_ribbon_synapse_counts():
    ihc_version = "ihc_counts_v6"
    synapse_dir = os.path.join(SYNAPSE_DIR_ROOT, ihc_version)
    tables = [entry.path for entry in os.scandir(synapse_dir) if "ihc_count_G_" in entry.name]
    syn_counts = []
    for tab in tables:
        x = pd.read_csv(tab, sep="\t")
        syn_counts.extend(x["synapse_count"].values.tolist())
    return syn_counts


def fig_06b(save_path, plot=False):
    """Box plot showing the counts for SGN and IHC per gerbil cochlea in comparison to literature values.
    """
    main_tick_size = 12
    main_label_size = 16

    rows = 1
    columns = 3

    fig, axes = plt.subplots(rows, columns, figsize=(columns*4, rows*4))
    ax = axes.flatten()

    sgn_values = [20050, 21995]
    ihc_values = [1100]

    ax[0].boxplot(sgn_values)
    ax[1].boxplot(ihc_values)

    # Labels and formatting
    ax[0].set_xticklabels(["SGN"], fontsize=main_label_size)

    ylim0 = 14000
    ylim1 = 30000
    ytick_gap = 4000
    y_ticks = [i for i in range((((ylim0 - 1) // ytick_gap) + 1) * ytick_gap, ylim1 + 1, ytick_gap)]

    ax[0].set_ylabel('Count per cochlea', fontsize=main_label_size)
    ax[0].set_yticks(y_ticks)
    ax[0].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax[0].set_ylim(ylim0, ylim1)

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    ax[0].set_xlim(xmin, xmax)
    lower_y, upper_y = literature_reference_values_gerbil("SGN")
    ax[0].hlines([lower_y, upper_y], xmin, xmax)
    ax[0].text(1, upper_y - 2000, "literature", color='C0', fontsize=main_tick_size, ha="center")
    ax[0].fill_between([xmin, xmax], lower_y, upper_y, color='C0', alpha=0.05, interpolate=True)

    ylim0 = 900
    ylim1 = 1400
    ytick_gap = 200
    y_ticks = [i for i in range((((ylim0 - 1) // ytick_gap) + 1) * ytick_gap, ylim1 + 1, ytick_gap)]

    ax[1].set_xticklabels(["IHC"], fontsize=main_label_size)

    ax[1].set_yticks(y_ticks)
    ax[1].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax[1].set_ylim(ylim0, ylim1)

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    ax[1].set_xlim(xmin, xmax)
    lower_y, upper_y = literature_reference_values_gerbil("IHC")
    ax[1].hlines([lower_y, upper_y], xmin, xmax)
    ax[1].fill_between([xmin, xmax], lower_y, upper_y, color='C0', alpha=0.05, interpolate=True)

    ribbon_synapse_counts = _load_ribbon_synapse_counts()
    ylim0 = -1
    ylim1 = 80
    ytick_gap = 20
    y_ticks = [i for i in range((((ylim0 - 1) // ytick_gap) + 1) * ytick_gap, ylim1 + 1, ytick_gap)]

    ax[2].boxplot(ribbon_synapse_counts)
    ax[2].set_xticklabels(["Ribbon Syn. per IHC"], fontsize=main_label_size)
    ax[2].set_yticks(y_ticks)
    ax[2].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax[2].set_ylim(ylim0, ylim1)

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    lower_y, upper_y = literature_reference_values_gerbil("synapse")
    ax[2].set_xlim(xmin, xmax)
    ax[2].hlines([lower_y, upper_y], xmin, xmax)
    ax[2].fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

    plt.tight_layout()
    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()
    else:
        plt.close()


def fig_06d(save_path, plot=False):
    """Plot the synapse distribution measured with different markers.

    The underlying measurements were done with 'scripts/measurements/synapse_colocalization.py'

    Here are the other relevant numbers for the analysis.
    Number of IHCs: 486
    Number of matched synapses: 3119
    Number and percentage of matched synapses for markers:
    CTBP2: 3119 / 3371 (92.52447345001484% matched)
    RibA : 3119 / 6701 (46.54529174750037% matched)
    """
    # TODO Plot this


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 6 of the cochlea paper.")
    parser.add_argument("figure_dir", type=str, help="Output directory for plots.", default="./panels")
    args = parser.parse_args()
    plot = False

    fig_06b(save_path=os.path.join(args.figure_dir, f"fig_06b.{FILE_EXTENSION}"), plot=plot)
    fig_06d(save_path=os.path.join(args.figure_dir, f"fig_06d.{FILE_EXTENSION}"), plot=plot)


if __name__ == "__main__":
    main()
