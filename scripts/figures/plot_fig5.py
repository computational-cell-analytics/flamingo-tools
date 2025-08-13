import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import literature_reference_values

png_dpi = 300


def _load_ribbon_synapse_counts():
    ihc_version = "ihc_counts_v4b"
    synapse_dir = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/{ihc_version}"
    tables = [entry.path for entry in os.scandir(synapse_dir) if "ihc_count_M_AMD" in entry.name]
    syn_counts = []
    for tab in tables:
        x = pd.read_csv(tab, sep="\t")
        syn_counts.extend(x["synapse_count"].values.tolist())
    return syn_counts


def fig_05c(save_path, plot=False):
    """Bar plot showing the distribution of synapse markers per IHC segmentation average over OTOF cochlea.
    """

    main_label_size = 20
    main_tick_size = 12
    htext_size = 10

    ribbon_synapse_counts = _load_ribbon_synapse_counts()

    fig, ax = plt.subplots(figsize=(8, 4))

    ylim0 = -1
    ylim1 = 24
    y_ticks = [i for i in range(0, 25, 5)]

    ax.set_ylabel("Ribbon Syn. per IHC", fontsize=main_label_size)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax.set_ylim(ylim0, ylim1)

    ax.boxplot(ribbon_synapse_counts)
    ax.set_xticklabels(["MOTOF1L"], fontsize=main_label_size)

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    lower_y, upper_y = literature_reference_values("synapse")
    ax.set_xlim(xmin, xmax)
    ax.hlines([lower_y, upper_y], xmin, xmax)
    ax.text(1.25, upper_y + 0.01*ax.get_ylim()[1]-ax.get_ylim()[0], "literature",
            color="C0", fontsize=htext_size, ha="center")
    ax.fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

    ihc_values = [14.1, 12.7, 13.8, 13.4]  # MLR226L, MLR226R, MLR227L, MLR227R

    ihc_value = np.mean(ihc_values)
    ihc_std = np.std(ihc_values)

    upper_y = ihc_value + 1.96 * ihc_std
    lower_y = ihc_value - 1.96 * ihc_std

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=["C1" for _ in range(2)])
    plt.text(1.25, upper_y + 0.01*ax.get_ylim()[1]-ax.get_ylim()[0], "healthy cochleae (95% confidence interval)",
             color="C1", fontsize=htext_size, ha="center")
    plt.fill_between([xmin, xmax], lower_y, upper_y, color="C1", alpha=0.05, interpolate=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)

    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 5 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig5")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: Monitoring of the Syn / IHC loss
    fig_05c(save_path=os.path.join(args.figure_dir, "fig_05c"), plot=args.plot)


if __name__ == "__main__":
    main()
