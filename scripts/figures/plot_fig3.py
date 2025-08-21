import argparse
import os
import imageio.v3 as imageio
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from util import sliding_runlength_sum, frequency_mapping, SYNAPSE_DIR_ROOT

INPUT_ROOT = "/home/pape/Work/my_projects/flamingo-tools/scripts/M_LR_000227_R/scale3/frequency_mapping"

png_dpi = 300


def _plot_colormap(vol, title, plot, save_path):
    # before creating the figure:
    matplotlib.rcParams.update({
        "font.size": 14,          # base font size
        "axes.titlesize": 18,     # for plt.title / ax.set_title
        "figure.titlesize": 18,   # for fig.suptitle (if you use it)
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    # Create the colormap
    fig, ax = plt.subplots(figsize=(6, 1.3))
    fig.subplots_adjust(bottom=0.5)

    freq_min = np.min(np.nonzero(vol))
    freq_max = vol.max()
    norm = colors.Normalize(vmin=freq_min, vmax=freq_max, clip=True)
    cmap = plt.get_cmap("viridis")

    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
    cb.set_label("Frequency [kHz]")
    plt.title(title)
    plt.tight_layout()
    if plot:
        plt.show()
    plt.savefig(save_path)
    plt.close()


def fig_03a(save_path, plot, plot_napari):
    path = os.path.join(INPUT_ROOT, "frequencies_IHC_v4c.tif")
    vol = imageio.imread(path)
    _plot_colormap(vol, title="Tonotopic Mapping: IHCs", plot=plot, save_path=save_path)

    # Show the image in napari for rendering.
    if plot_napari:
        import napari
        v = napari.Viewer()
        v.add_image(vol, colormap="viridis")
        napari.run()


def fig_03b(save_path, plot, plot_napari):
    path = os.path.join(INPUT_ROOT, "frequencies_SGN_v2.tif")
    vol = imageio.imread(path)
    _plot_colormap(vol, title="Tonotopic Mapping: SGNs", plot=plot, save_path=save_path)

    # Show the image in napari for rendering.
    if plot_napari:
        import napari
        v = napari.Viewer()
        v.add_image(vol, colormap="viridis")


def fig_03c_rl(save_path, plot=False):
    tables = glob("./ihc_counts/ihc_count_M_LR*.tsv")
    fig, ax = plt.subplots(figsize=(8, 4))

    width = 50  # micron

    for tab_path in tables:
        # TODO map to alias
        alias = os.path.basename(tab_path)[10:-4].replace("_", "").replace("0", "")
        tab = pd.read_csv(tab_path, sep="\t")
        run_length = tab["run_length"].values
        syn_count = tab["synapse_count"].values

        # Compute the running sum of 10 micron.
        run_length, syn_count_running = sliding_runlength_sum(run_length, syn_count, width=width)
        ax.plot(run_length, syn_count_running, label=alias)

    ax.set_xlabel("Length [µm]")
    ax.set_ylabel("Synapse Count")
    ax.set_title(f"Ribbon Syn. per IHC: Runnig sum @ {width} µm")
    ax.legend(title="cochlea")
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    if plot:
        plt.show()
    else:
        plt.close()


def fig_03c_octave(save_path, plot=False):
    # TODO update this table
    ihc_version = "ihc_counts_v4c"
    tables = glob(os.path.join(SYNAPSE_DIR_ROOT, ihc_version, "ihc_count_M_LR*.tsv"))

    result = {"cochlea": [], "octave_band": [], "value": []}
    for tab_path in tables:
        # TODO map to alias
        alias = os.path.basename(tab_path)[10:-4].replace("_", "").replace("0", "")
        tab = pd.read_csv(tab_path, sep="\t")
        freq = tab["frequency"].values
        syn_count = tab["synapse_count"].values

        # Compute the running sum of 10 micron.
        octave_binned = frequency_mapping(freq, syn_count, animal="mouse")

        result["cochlea"].extend([alias] * len(octave_binned))
        result["octave_band"].extend(octave_binned.axes[0].values.tolist())
        result["value"].extend(octave_binned.values.tolist())

    result = pd.DataFrame(result)
    bin_labels = pd.unique(result["octave_band"])
    band_to_x = {band: i for i, band in enumerate(bin_labels)}
    result["x_pos"] = result["octave_band"].map(band_to_x)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, grp in result.groupby("cochlea"):
        ax.scatter(grp["x_pos"], grp["value"], label=name, s=60, alpha=0.8)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Octave band (kHz)")

    ax.set_ylabel("Average Ribbon Synapse Count per IHC")
    ax.set_title("Ribbon synapse count per octave band")
    ax.legend(title="Cochlea")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    if plot:
        plt.show()
    else:
        plt.close()


def fig_03d(save_path, plot, print_stats=True):
    result_folder = "../measurements/subtype_analysis"
    files = glob(os.path.join(result_folder, "*.tsv"))

    for ff in files:
        fname = os.path.basename(ff)
        cochlea = fname[:-len("_subtype_analysis.tsv")]
        table = pd.read_csv(ff, sep="\t")

        subtype_table = table[[col for col in table.columns if col.startswith("is_")]]
        assert subtype_table.shape[1] == 2
        n_sgns = len(subtype_table)

        if print_stats:
            print(cochlea)
            for col in subtype_table.columns:
                vals = table[col].values
                subtype = col[3:]
                n_subtype = vals.sum()
                print(subtype, ":", n_subtype, "/", n_sgns, f"({np.round(float(n_subtype) / n_sgns * 100, 2)} %)")

            coexpr = np.logical_and(subtype_table.iloc[:, 0].values, subtype_table.iloc[:, 1].values)
            print("Co-expression:", coexpr.sum())


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 3 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig3")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel A: Tonotopic mapping of IHCs (rendering in napari)
    # fig_03a(save_path=os.path.join(args.figure_dir, "fig_03a.png"), plot=args.plot, plot_napari=False)

    # Panel B: Tonotopic mapping of SGNs (rendering in napari)
    # fig_03b(save_path=os.path.join(args.figure_dir, "fig_03b.png"), plot=args.plot, plot_napari=False)

    # Panel C: Spatial distribution of synapses across the cochlea.
    # We have two options: running sum over the runlength or per octave band
    # fig_03c_rl(save_path=os.path.join(args.figure_dir, "fig_03c_runlength.png"), plot=args.plot)
    # fig_03c_octave(save_path=os.path.join(args.figure_dir, "fig_03c_octave.png"), plot=args.plot)

    # Panel D: Spatial distribution of SGN sub-types.
    fig_03d(save_path=os.path.join(args.figure_dir, "fig_03d.png"), plot=args.plot)


if __name__ == "__main__":
    main()
