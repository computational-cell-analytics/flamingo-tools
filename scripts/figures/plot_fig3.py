import argparse
import os
import imageio.v3 as imageio
from glob import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from util import sliding_runlength_sum, frequency_mapping, SYNAPSE_DIR_ROOT, to_alias

INPUT_ROOT = "/home/pape/Work/my_projects/flamingo-tools/scripts/M_LR_000227_R/scale3"

TYPE_TO_CHANNEL = {
    "Type-Ia": "CR",
    "Type-Ib": "Calb1",
    "Type-Ic": "Lypd1",
    "Type-II": "Prph",
}

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
    path_ihc = os.path.join(INPUT_ROOT, "frequencies_IHC_v4c.tif")
    path_sgn = os.path.join(INPUT_ROOT, "frequencies_SGN_v2.tif")
    sgn = imageio.imread(path_sgn)
    ihc = imageio.imread(path_ihc)
    _plot_colormap(sgn, title="Tonotopic Mapping", plot=plot, save_path=save_path)

    # Show the image in napari for rendering.
    if plot_napari:
        import napari
        v = napari.Viewer()
        v.add_image(ihc, colormap="viridis")
        v.add_image(sgn, colormap="viridis")
        napari.run()


def fig_03c_rl(save_path, plot=False):
    ihc_version = "ihc_counts_v4c"
    synapse_dir = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/{ihc_version}"
    tables = [entry.path for entry in os.scandir(synapse_dir) if "ihc_count_M_LR" in entry.name]

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
    ihc_version = "ihc_counts_v4c"
    tables = glob(os.path.join(SYNAPSE_DIR_ROOT, ihc_version, "ihc_count_M_LR*.tsv"))
    assert len(tables) == 4, len(tables)

    result = {"cochlea": [], "octave_band": [], "value": []}
    for tab_path in tables:
        cochlea = Path(tab_path).stem.lstrip("ihc_count")
        alias = to_alias(cochlea)
        tab = pd.read_csv(tab_path, sep="\t")
        freq = tab["frequency"].values
        syn_count = tab["synapse_count"].values

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
    plt.legend(title="Cochlea")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    if plot:
        plt.show()
    else:
        plt.close()


def fig_03d_fraction(save_path, plot):
    result_folder = "../measurements/subtype_analysis"
    files = glob(os.path.join(result_folder, "*.tsv"))

    # FIXME
    analysis = {
        "M_AMD_N62_L": ["CR", "Calb1"],
        "M_LR_000214_L": ["CR"],
    }

    results = {"type": [], "fraction": [], "cochlea": []}
    for ff in files:
        fname = os.path.basename(ff)
        cochlea = fname[:-len("_subtype_analysis.tsv")]

        if cochlea not in analysis:
            continue

        table = pd.read_csv(ff, sep="\t")

        subtype_table = table[[col for col in table.columns if col.startswith("is_")]]
        assert subtype_table.shape[1] == 2
        n_sgns = len(subtype_table)

        print(cochlea)
        for col in subtype_table.columns:
            vals = table[col].values
            subtype = col[3:]
            channel = TYPE_TO_CHANNEL[subtype]
            if channel not in analysis[cochlea]:
                continue
            n_subtype = vals.sum()
            subtype_fraction = np.round(float(n_subtype) / n_sgns * 100, 2)
            name = f"{subtype} ({channel})"
            print("{name}:", n_subtype, "/", n_sgns, f"({subtype_fraction} %)")

            results["type"].append(name)
            results["fraction"].append(subtype_fraction)
            results["cochlea"].append(cochlea)

        # coexpr = np.logical_and(subtype_table.iloc[:, 0].values, subtype_table.iloc[:, 1].values)
        # print("Co-expression:", coexpr.sum())

    results = pd.DataFrame(results)
    fig, ax = plt.subplots()
    for cochlea, group in results.groupby("cochlea"):
        ax.scatter(group["type"], group["fraction"], label=cochlea)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Type")
    ax.legend(title="Cochlea ID")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    if plot:
        plt.show()
    else:
        plt.close()


# TODO
def fig_03d_octave(save_path, plot):
    pass


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 3 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig3")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel A: Tonotopic mapping of SGNs and IHCs (rendering in napari + heatmap)
    fig_03a(save_path=os.path.join(args.figure_dir, "fig_03a_cmap.png"), plot=args.plot, plot_napari=True)

    # Panel C: Spatial distribution of synapses across the cochlea.
    # We have two options: running sum over the runlength or per octave band
    fig_03c_rl(save_path=os.path.join(args.figure_dir, "fig_03c_runlength.png"), plot=args.plot)
    fig_03c_octave(save_path=os.path.join(args.figure_dir, "fig_03c_octave.png"), plot=args.plot)

    # Panel D: Spatial distribution of SGN sub-types.
    fig_03d_fraction(save_path=os.path.join(args.figure_dir, "fig_03d_fraction.png"), plot=args.plot)
    fig_03d_octave(save_path=os.path.join(args.figure_dir, "fig_03d_octave.png"), plot=args.plot)


if __name__ == "__main__":
    main()
