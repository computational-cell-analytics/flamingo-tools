import argparse
import os

import matplotlib.pyplot as plt

from util import literature_reference_values_gerbil

png_dpi = 300


def fig_06a(save_path, plot=False):
    """Box plot showing the counts for SGN and IHC per gerbil cochlea in comparison to literature values.
    """
    main_tick_size = 12
    main_label_size = 16

    rows = 1
    columns = 2

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
    ax[0].text(1, upper_y + 100, "literature reference", color='C0', fontsize=main_tick_size, ha="center")
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
    ax[1].text(1, upper_y + 10, "literature reference", color='C0', fontsize=main_tick_size, ha="center")
    ax[1].fill_between([xmin, xmax], lower_y, upper_y, color='C0', alpha=0.05, interpolate=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=png_dpi)
    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 2 of the cochlea paper.")
    parser.add_argument("figure_dir", type=str, help="Output directory for plots.", default="./panels")
    args = parser.parse_args()
    plot = False

    fig_06a(save_path=os.path.join(args.figure_dir, "fig_06a"), plot=plot)


if __name__ == "__main__":
    main()
