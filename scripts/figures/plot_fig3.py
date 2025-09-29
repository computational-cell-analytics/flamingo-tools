import argparse
import json
import os
import pickle

import imageio.v3 as imageio
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target
from util import sliding_runlength_sum, frequency_mapping, SYNAPSE_DIR_ROOT

INPUT_ROOT = "/home/martin/Documents/lightsheet-cochlea/M_LR_000227_R"

TYPE_TO_CHANNEL = {
    "Type-Ia": "CR",
    "Type-Ib": "Calb1",
    "Type-Ic": "Lypd1",
    "Type-II": "Prph",
}

FILE_EXTENSION = "png"

png_dpi = 300

# The cochlea for the CHReef analysis.
COCHLEAE_DICT = {
    "M_LR_000226_L": {"alias": "M01L", "component": [1]},
    "M_LR_000226_R": {"alias": "M01R", "component": [1]},
    "M_LR_000227_L": {"alias": "M02L", "component": [1]},
    "M_LR_000227_R": {"alias": "M02R", "component": [1]},
}


def get_tonotopic_data():
    s3 = create_s3_target()
    source_name = "IHC_v4c"
    ihc_version = source_name.split("_")[1]
    cache_path = "./tonotopic_data.pkl"
    cochleae = [key for key in COCHLEAE_DICT.keys()]

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    chreef_data = {}
    for cochlea in cochleae:
        print("Processsing cochlea:", cochlea)
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the seg table and filter the compartments.
        source = sources[source_name]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        # May need to be adjusted for some cochleae.
        component_labels = COCHLEAE_DICT[cochlea]["component"]
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        ihc_dir = f"ihc_counts_{ihc_version}"
        synapse_dir = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/{ihc_dir}"
        tab_path = os.path.join(synapse_dir, f"ihc_count_{cochlea}.tsv")
        syn_tab = pd.read_csv(tab_path, sep="\t")
        syn_ids = syn_tab["label_id"].values

        syn_per_ihc = [0 for _ in range(len(table))]
        table.loc[:, "syn_per_IHC"] = syn_per_ihc
        for syn_id in syn_ids:
            table.loc[table["label_id"] == syn_id, "syn_per_IHC"] = syn_tab.at[syn_tab.index[syn_tab["label_id"] == syn_id][0], "synapse_count"]  # noqa

        # The relevant values for analysis.
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "syn_per_IHC"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skippped.")
            continue

        chreef_data[cochlea] = values

    with open(cache_path, "wb") as f:
        pickle.dump(chreef_data, f)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _plot_colormap(vol, title, plot, save_path, cmap="viridis"):
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
    # norm = colors.Normalize(vmin=freq_min, vmax=freq_max, clip=True)
    norm = colors.LogNorm(vmin=freq_min, vmax=freq_max, clip=True)
    tick_values = np.array([10, 20, 40, 80])

    cmap = plt.get_cmap(cmap)

    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal",
                      ticks=tick_values)
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_label("Frequency [kHz]")
    plt.title(title)
    plt.tight_layout()
    if plot:
        plt.show()

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def fig_03a(save_path, plot, plot_napari, cmap="viridis"):
    path_ihc = os.path.join(INPUT_ROOT, "frequencies_IHC_v4c.tif")
    path_sgn = os.path.join(INPUT_ROOT, "frequencies_SGN_v2.tif")
    sgn = imageio.imread(path_sgn)
    ihc = imageio.imread(path_ihc)
    _plot_colormap(sgn, title="Tonotopic Mapping", plot=plot, save_path=save_path, cmap=cmap)

    # Show the image in napari for rendering.
    if plot_napari:
        import napari
        from napari.utils import Colormap
        # cmap = plt.get_cmap(cmap)
        mpl_cmap = plt.get_cmap(cmap)

        # Sample it into an array of RGBA values
        colors = mpl_cmap(np.linspace(0, 1, 256))

        # Wrap into napari Colormap
        napari_cmap = Colormap(colors, name=f"{cmap}_custom")

        v = napari.Viewer()
        v.add_image(ihc, colormap=napari_cmap)
        v.add_image(sgn, colormap=napari_cmap)
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

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_03c_octave(tonotopic_data, save_path, plot=False, use_alias=True):
    ihc_version = "ihc_counts_v4c"
    tables = glob(os.path.join(SYNAPSE_DIR_ROOT, ihc_version, "ihc_count_M_LR*.tsv"))
    assert len(tables) == 4, len(tables)

    result = {"cochlea": [], "octave_band": [], "value": []}
    for name, values in tonotopic_data.items():
        if use_alias:
            alias = COCHLEAE_DICT[name]["alias"]
        else:
            alias = name.replace("_", "").replace("0", "")

        freq = values["frequency[kHz]"].values
        syn_count = values["syn_per_IHC"].values
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

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

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

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

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
    tonotopic_data = get_tonotopic_data()

    # Panel A: Tonotopic mapping of SGNs and IHCs (rendering in napari + heatmap)
    cmap = "plasma"
    fig_03a(save_path=os.path.join(args.figure_dir, f"fig_03a_cmap_{cmap}.{FILE_EXTENSION}"),
            plot=args.plot, plot_napari=True, cmap=cmap)

    # Panel C: Spatial distribution of synapses across the cochlea.
    # We have two options: running sum over the runlength or per octave band
    # fig_03c_rl(save_path=os.path.join(args.figure_dir, f"fig_03c_runlength.{FILE_EXTENSION}"), plot=args.plot)
    fig_03c_octave(tonotopic_data=tonotopic_data,
                    save_path=os.path.join(args.figure_dir, f"fig_03c_octave.{FILE_EXTENSION}"),
                    plot=args.plot)

    # Panel D: Spatial distribution of SGN sub-types.
    # fig_03d_fraction(save_path=os.path.join(args.figure_dir, f"fig_03d_fraction.{FILE_EXTENSION}"), plot=args.plot)
    # fig_03d_octave(save_path=os.path.join(args.figure_dir, f"fig_03d_octave.{FILE_EXTENSION}"), plot=args.plot)


if __name__ == "__main__":
    main()
