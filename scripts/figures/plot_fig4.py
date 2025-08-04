import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import frequency_mapping

INTENSITY_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/measurements"  # noqa

# The cochlea for the CHReef analysis.
COCHLEAE = [
    "M_LR_000143_L",
    "M_LR_000144_L",
    "M_LR_000145_L",
    "M_LR_000153_L",
    "M_LR_000155_L",
    "M_LR_000189_L",
    "M_LR_000143_R",
    "M_LR_000144_R",
    "M_LR_000145_R",
    "M_LR_000153_R",
    "M_LR_000155_R",
    "M_LR_000189_R",
]

png_dpi = 300


def get_chreef_data():
    s3 = create_s3_target()
    source_name = "SGN_v2"

    cache_path = "./chreef_data.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    chreef_data = {}
    for cochlea in COCHLEAE:
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
        table = table[table.component_labels == 1]
        # The relevant values for analysis.
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "marker_labels"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skippped.")
            continue

        fname = f"{cochlea.replace('_', '-')}_GFP_SGN-v2_object-measures.tsv"
        intensity_file = os.path.join(INTENSITY_ROOT, fname)
        assert os.path.exists(intensity_file), intensity_file
        intensity_table = pd.read_csv(intensity_file, sep="\t")
        values = values.merge(intensity_table, on="label_id")

        chreef_data[cochlea] = values

    with open(cache_path, "wb") as f:
        chreef_data = pickle.dump(chreef_data, f)
    return chreef_data


def group_lr(names_lr, values):
    assert len(names_lr) == len(values)
    names = []
    values_left, values_right = {}, {}
    for name_lr, val in zip(names_lr, values):
        name, side = name_lr[:-1], name_lr[-1]
        names.append(name)
        if side == "R":
            values_right[name] = val
        elif side == "L":
            values_left[name] = val
        else:
            raise RuntimeError
    names = sorted(list(set(names)))

    values_left = [values_left.get(name, np.nan) for name in names]
    values_right = [values_right.get(name, np.nan) for name in names]

    return names, values_left, values_right


def fig_04c(chreef_data, save_path, plot=False, plot_by_side=False):
    """Box plot showing the SGN counts of ChReef treated cochleae compared to untreated ones.
    """
    # Previous version with hard-coded values.
    # cochlea = ["M_LR_000144_L", "M_LR_000145_L", "M_LR_000151_R"]
    # alias = ["c01", "c02", "c03"]
    # sgns = [7796, 6119, 9225]

    # TODO map the cochlea name to its alias
    alias = [name.replace("_", "").replace("0", "") for name in chreef_data.keys()]
    sgns = [len(vals) for vals in chreef_data.values()]

    if plot_by_side:
        alias, sgns_left, sgns_right = group_lr(alias, sgns)

    x = np.arange(len(alias))

    # Plot
    plt.figure(figsize=(8, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 12
    legendsize = 16

    if plot_by_side:
        plt.scatter(x, sgns_left, label="SGN count (Left)", marker="o", s=80)
        plt.scatter(x, sgns_right, label="SGN count (Right)", marker="x", s=80)
    else:
        plt.scatter(x, sgns, label="SGN count", marker="o", s=80)

    # Labels and formatting
    plt.xticks(x, alias, fontsize=sub_label_size)
    plt.xlabel("Cochlea", fontsize=main_label_size)
    plt.yticks(fontsize=main_tick_size)
    plt.ylabel("SGN count per cochlea", fontsize=main_label_size)
    plt.ylim(4000, 13800)
    plt.legend(loc="best", fontsize=sub_label_size)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11),
               ncol=3, fancybox=True, shadow=False, framealpha=0.8, fontsize=legendsize)

    # set range of literature values
    xmin = -0.5
    xmax = len(alias) - 0.5
    plt.xlim(xmin, xmax)
    upper_y = 12000
    lower_y = 10000
    plt.hlines([lower_y, upper_y], xmin, xmax)
    plt.text(1, lower_y - 400, "literature reference (WIP)", color="C0", fontsize=main_tick_size, ha="center")
    plt.fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

    sgn_values = [11153, 11398, 10333, 11820]
    sgn_value = np.mean(sgn_values)
    sgn_std = np.std(sgn_values)

    upper_y = sgn_value + 1.96 * sgn_std
    lower_y = sgn_value - 1.96 * sgn_std

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=["C1" for _ in range(2)])
    plt.text(1, upper_y + 100, "untreated cochleae (95% confidence interval)",
             color="C1", fontsize=main_tick_size, ha="center")
    plt.fill_between([xmin, xmax], lower_y, upper_y, color="C1", alpha=0.05, interpolate=True)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        plt.close()


def fig_04d(chreef_data, save_path, plot=False, plot_by_side=False, intensity=False):
    """Transduction efficiency per cochlea.
    """
    # TODO map the cochlea name to its alias
    alias = [name.replace("_", "").replace("0", "") for name in chreef_data.keys()]

    values = []
    for vals in chreef_data.values():
        if intensity:
            intensities = vals["median"].values
            values.append(intensities.mean())
        else:
            # The marker labels don't make sense yet, they are in
            # 0: unlabeled
            # 1: positive
            # 2: negative
            # but they should all be either positive or negative.
            # Or am I missing something?
            marker_labels = vals["marker_labels"].values
            n_pos = (marker_labels == 1).sum()
            n_neg = (marker_labels == 2).sum()
            eff = float(n_pos) / (n_pos + n_neg)
            values.append(eff)

    if plot_by_side:
        alias, values_left, values_right = group_lr(alias, values)

    x = np.arange(len(alias))

    # Plot
    plt.figure(figsize=(8, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 12
    legendsize = 16

    label = "Intensity" if intensity else "Transduction efficiency"
    if plot_by_side:
        plt.scatter(x, values_left, label=f"{label} (Left)", marker="o", s=80)
        plt.scatter(x, values_right, label=f"{label} (Right)", marker="x", s=80)
    else:
        plt.scatter(x, values, label=label, marker="o", s=80)

    # Labels and formatting
    plt.xticks(x, alias, fontsize=sub_label_size)
    plt.xlabel("Cochlea", fontsize=main_label_size)
    plt.yticks(fontsize=main_tick_size)
    plt.ylabel("Transduction efficiency", fontsize=main_label_size)
    plt.legend(loc="best", fontsize=sub_label_size)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11),
               ncol=3, fancybox=True, shadow=False, framealpha=0.8, fontsize=legendsize)
    if not intensity:
        plt.ylim(0.5, 1.05)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        plt.close()


def fig_04e(chreef_data, save_path, plot, intensity=False):

    result = {"cochlea": [], "octave_band": [], "value": []}
    for name, values in chreef_data.items():
        # TODO map name to alias
        alias = name.replace("_", "").replace("0", "")

        freq = values["frequency[kHz]"].values
        if intensity:
            intensity_values = values["median"].values
            octave_binned = frequency_mapping(freq, intensity_values, animal="mouse")
        else:
            marker_labels = values["marker_labels"].values
            octave_binned = frequency_mapping(freq, marker_labels, animal="mouse", transduction_efficiency=True)

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
    if intensity:
        ax.set_ylabel("Marker Intensity")
        ax.set_title("Median intensity per octave band")
    else:
        ax.set_ylabel("Transduction Efficiency")
        ax.set_ylim(0.5, 1.05)
        ax.set_title("Transduction per octave band")
    ax.legend(title="Cochlea")
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 4 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig4")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Get the chreef data as a dictionary of cochlea name to measurements.
    chreef_data = get_chreef_data()
    # M_LR_00143_L is a complete outlier
    chreef_data.pop("M_LR_000143_L")

    # Create the panels.
    fig_04c(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04c"), plot=args.plot, plot_by_side=False)

    fig_04d(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04d_transduction"), plot=args.plot, plot_by_side=True)  # noqa
    fig_04d(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04d_intensity"), plot=args.plot, plot_by_side=True, intensity=True)  # noqa

    fig_04e(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04e_transduction"), plot=args.plot)
    fig_04e(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04e_intensity"), plot=args.plot, intensity=True)


if __name__ == "__main__":
    main()
