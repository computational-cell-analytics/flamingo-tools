import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import frequency_mapping  # , literature_reference_values

INTENSITY_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/measurements2"  # noqa

# The cochlea for the CHReef analysis.
COCHLEAE_DICT = {
    "M_LR_000143_L": {"alias": "M0L", "component": [1]},
    "M_LR_000144_L": {"alias": "M05L", "component": [1]},
    "M_LR_000145_L": {"alias": "M06L", "component": [1]},
    "M_LR_000153_L": {"alias": "M07L", "component": [1]},
    "M_LR_000155_L": {"alias": "M08L", "component": [1, 2, 3]},
    "M_LR_000189_L": {"alias": "M09L", "component": [1]},
    "M_LR_000143_R": {"alias": "M0R", "component": [1]},
    "M_LR_000144_R": {"alias": "M05R", "component": [1]},
    "M_LR_000145_R": {"alias": "M06R", "component": [1]},
    "M_LR_000153_R": {"alias": "M07R", "component": [1]},
    "M_LR_000155_R": {"alias": "M08R", "component": [1]},
    "M_LR_000189_R": {"alias": "M09R", "component": [1]},
    "G_EK_000049_L": {"alias": "G1L", "component": [1, 3, 4, 5]},
    "G_EK_000049_R": {"alias": "G1R", "component": [1, 2]},
}

png_dpi = 300


def get_chreef_data(animal="mouse"):
    s3 = create_s3_target()
    source_name = "SGN_v2"

    if animal == "mouse":
        cache_path = "./chreef_data.pkl"
        cochleae = [key for key in COCHLEAE_DICT.keys() if "M_" in key]
    else:
        cache_path = "./chreef_data_gerbil.pkl"
        cochleae = [key for key in COCHLEAE_DICT.keys() if "G_" in key]

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
        pickle.dump(chreef_data, f)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


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


def fig_04c(chreef_data, save_path, plot=False, plot_by_side=False, use_alias=True):
    """Box plot showing the SGN counts of ChReef treated cochleae compared to healthy ones.
    """
    # Previous version with hard-coded values.
    # cochlea = ["M_LR_000144_L", "M_LR_000145_L", "M_LR_000151_R"]
    # alias = ["c01", "c02", "c03"]
    # sgns = [7796, 6119, 9225]

    # TODO have central function for alias for all plots?
    if use_alias:
        alias = [COCHLEAE_DICT[k]["alias"] for k in chreef_data.keys()]
    else:
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

    xmin = -0.5
    xmax = len(alias) - 0.5
    plt.xlim(xmin, xmax)

    # set range of literature values
    # lower_y, upper_y = literature_reference_values("SGN")
    # plt.hlines([lower_y, upper_y], xmin, xmax)
    # plt.text(1.5, lower_y - 400, "literature", color="C0", fontsize=main_tick_size, ha="center")
    # plt.fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

    sgn_values = [11153, 11398, 10333, 11820]
    sgn_value = np.mean(sgn_values)
    sgn_std = np.std(sgn_values)

    upper_y = sgn_value + 1.96 * sgn_std
    lower_y = sgn_value - 1.96 * sgn_std

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=["C1" for _ in range(2)])
    plt.text(1.5, upper_y + 100, "healthy cochleae (95% confidence interval)",
             color="C1", fontsize=main_tick_size, ha="center")
    plt.fill_between([xmin, xmax], lower_y, upper_y, color="C1", alpha=0.05, interpolate=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_04d(chreef_data, save_path, plot=False, plot_by_side=False, intensity=False, gerbil=False, use_alias=True):
    """Transduction efficiency per cochlea.
    """
    if use_alias:
        alias = [COCHLEAE_DICT[k]["alias"] for k in chreef_data.keys()]
    else:
        alias = [name.replace("_", "").replace("0", "") for name in chreef_data.keys()]

    values = []
    for vals in chreef_data.values():
        if intensity:
            intensities = vals["median"].values
            values.append(intensities.mean())
        else:
            # marker labels
            # 0: unlabeled - no median intensity in object_measures table
            # 1: positive
            # 2: negative
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
    legendsize = 16 if intensity else 12

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
    plt.ylabel(label, fontsize=main_label_size)
    plt.legend(loc="best", fontsize=sub_label_size)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11),
               ncol=3, fancybox=True, shadow=False, framealpha=0.8, fontsize=legendsize)
    if not intensity:
        if gerbil:
            plt.ylim(0.3, 1.05)
        else:
            plt.ylim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_04e(chreef_data, save_path, plot, intensity=False, gerbil=False, use_alias=True):

    result = {"cochlea": [], "octave_band": [], "value": []}
    for name, values in chreef_data.items():
        if use_alias:
            alias = COCHLEAE_DICT[name]["alias"]
        else:
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

    sub_tick_label_size = 8
    tick_label_size = 12
    label_size = 12
    legend_size = 8
    if intensity:
        band_label_offset_y = 0.09
    else:
        band_label_offset_y = 0.07
        if gerbil:
            ax.set_ylim(0.05, 1.05)
        else:
            ax.set_ylim(0.45, 1.05)

    # Offsets within each octave band
    offset_map = {"L": -0.15, "R": 0.15}
    sublabels = {"L": "L", "R": "R"}

    # Assign a color to each cochlea (ignoring side)
    cochleas = sorted({name_lr[:-1] for name_lr in result["cochlea"].unique()})
    colors = plt.cm.tab10.colors  # pick a colormap
    color_map = {cochlea: colors[i % len(colors)] for i, cochlea in enumerate(cochleas)}

    # Track which cochlea names we have already added to the legend
    legend_added = set()

    all_x_positions = []
    all_x_labels = []

    for name_lr, grp in result.groupby("cochlea"):
        name, side = name_lr[:-1], name_lr[-1]
        x_positions = grp["x_pos"] + offset_map[side]
        ax.scatter(
            x_positions,
            grp["value"],
            label=name if name not in legend_added else None,
            s=60,
            alpha=0.8,
            marker="o" if side == "L" else "x",
            color=color_map[name]
        )
        if name not in legend_added:
            legend_added.add(name)

        # Store for sublabel ticks
        all_x_positions.extend(x_positions)
        all_x_labels.extend([sublabels[side]] * len(x_positions))

    # Create combined tick positions & labels
    main_ticks = range(len(bin_labels))
    # add a final tick for label '>64k'
    ax.set_xticks([pos + offset_map["L"] for pos in main_ticks[:-1]] +
                  [pos + offset_map["R"] for pos in main_ticks[:-1]] +
                  [pos for pos in main_ticks[-1:]])
    ax.set_xticklabels(["L"] * len(main_ticks[:-1]) + ["R"] * len(main_ticks[:-1]) + [""], fontsize=sub_tick_label_size)

    # Add main octave band labels above sublabels
    for i, label in enumerate(bin_labels):
        ax.text(i, ax.get_ylim()[0] - band_label_offset_y*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                label, ha='center', va='top', fontsize=tick_label_size, fontweight='bold')

    ax.set_xlabel("Octave band (kHz)", fontsize=label_size)
    ax.xaxis.set_label_coords(.5, -.16)

    if intensity:
        ax.set_ylabel("Marker Intensity", fontsize=label_size)
        ax.set_title("Intensity per octave band (Left/Right)")
    else:
        ax.set_ylabel("Transduction Efficiency", fontsize=label_size)
        ax.set_title("Transduction efficiency per octave band (Left/Right)")

    ax.legend(title="Cochlea", fontsize=legend_size)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 4 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig4")
    parser.add_argument("--no_alias", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    use_alias = not args.no_alias
    os.makedirs(args.figure_dir, exist_ok=True)

    # Get the chreef data as a dictionary of cochlea name to measurements.
    chreef_data = get_chreef_data()
    # M_LR_00143_L is a complete outlier
    chreef_data.pop("M_LR_000143_L")
    # remove other cochlea to have only pairs remaining
    chreef_data.pop("M_LR_000143_R")

    # Create the panels:

    # C: The SGN count compared to reference values from literature and healthy
    # Maybe remove literature reference from plot?
    fig_04c(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04c"),
            plot=args.plot, plot_by_side=True, use_alias=use_alias)

    # D: The transduction efficiency. We also plot GFP intensities.
    fig_04d(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04d_transduction"),
            plot=args.plot, plot_by_side=True, use_alias=use_alias)
    fig_04d(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04d_intensity"),
            plot=args.plot, plot_by_side=True, intensity=True, use_alias=use_alias)

    fig_04e(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04e_transduction"),
            plot=args.plot, use_alias=use_alias)
    fig_04e(chreef_data, save_path=os.path.join(args.figure_dir, "fig_04e_intensity"),
            plot=args.plot, intensity=True, use_alias=use_alias)

    chreef_data_gerbil = get_chreef_data(animal="gerbil")
    fig_04d(chreef_data_gerbil, save_path=os.path.join(args.figure_dir, "fig_04d_gerbil_transduction"),
            plot=args.plot, plot_by_side=True, gerbil=True, use_alias=use_alias)
    fig_04d(chreef_data_gerbil, save_path=os.path.join(args.figure_dir, "fig_04d_gerbil_intensity"),
            plot=args.plot, plot_by_side=True, intensity=True, use_alias=use_alias)

    fig_04e(chreef_data_gerbil, save_path=os.path.join(args.figure_dir, "fig_04e_gerbil_transduction"),
            plot=args.plot, gerbil=True, use_alias=use_alias)
    fig_04e(chreef_data_gerbil, save_path=os.path.join(args.figure_dir, "fig_04e_gerbil_intensity"),
            plot=args.plot, intensity=True, use_alias=use_alias)


if __name__ == "__main__":
    main()
