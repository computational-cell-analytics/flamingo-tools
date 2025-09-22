import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import frequency_mapping, prism_style, prism_cleanup_axes  # , literature_reference_values

# from statsmodels.nonparametric.smoothers_lowess import lowess

INTENSITY_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/measurements2"  # noqa

# The cochlea for the CHReef analysis.
COCHLEAE_DICT = {
    "M_LR_000143_L": {"alias": "M0L", "component": [1]},
    "M_LR_000144_L": {"alias": "M05L", "component": [1]},
    "M_LR_000145_L": {"alias": "M06L", "component": [1]},
    "M_LR_000153_L": {"alias": "M07L", "component": [1, 2, 3]},
    "M_LR_000155_L": {"alias": "M08L", "component": [1]},
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

FILE_EXTENSION = "png"
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
    prism_style()

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
    fig, ax = plt.subplots(figsize=(5, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16
    legendsize = 12

    if plot_by_side:
        plt.scatter(x, sgns_left, label="Injected", marker="o", s=80)
        plt.scatter(x, sgns_right, label="Non-Injected", marker="x", s=80)
    else:
        plt.scatter(x, sgns, label="SGN count", marker="o", s=80)

    # Labels and formatting
    plt.xticks(x, alias, fontsize=sub_label_size)
    plt.yticks(fontsize=main_tick_size)
    plt.ylabel("SGN count per cochlea", fontsize=main_label_size)
    plt.ylim(4000, 15800)
    plt.legend(loc="upper right", fontsize=legendsize)

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
    plt.text(2, upper_y + 200, "untreated cochleae\n(95% confidence interval)",
             color="C1", fontsize=14, ha="center")
    plt.fill_between([xmin, xmax], lower_y, upper_y, color="C1", alpha=0.05, interpolate=True)

    plt.tight_layout()

    prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_04d(chreef_data, save_path, plot=False, plot_by_side=False, intensity=False, gerbil=False, use_alias=True):
    """Transduction efficiency per cochlea.
    """
    prism_style()
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
    fig, ax = plt.subplots(figsize=(5, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16
    legendsize = 12

    label = "Intensity" if intensity else "Transduction efficiency"

    if plot_by_side:
        plt.scatter(x, values_left, label="Injected", marker="o", s=80)
        plt.scatter(x, values_right, label="Non-Injected", marker="x", s=80)
    else:
        plt.scatter(x, values, label=label, marker="o", s=80)

    # Labels and formatting
    plt.xticks(x, alias, fontsize=sub_label_size)
    plt.yticks(fontsize=main_tick_size)
    plt.ylabel(label, fontsize=main_label_size)
    plt.legend(loc="upper right", fontsize=legendsize)
    if not intensity:
        if gerbil:
            plt.ylim(0.3, 1.05)
        else:
            plt.ylim(0.5, 1.05)

    plt.tight_layout()
    prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_04e(chreef_data, save_path, plot, intensity=False, gerbil=False, use_alias=True, trendlines=False):
    prism_style()

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
        band_label_offset_y = 0.09
        if gerbil:
            ax.set_ylim(0.05, 1.05)
        else:
            ax.set_ylim(0.45, 1.05)

    # Offsets within each octave band
    offset_map = {"L": -0.15, "R": 0.15}

    # Assign a color to each cochlea (ignoring side)
    cochleas = sorted({name_lr[:-1] for name_lr in result["cochlea"].unique()})
    colors = plt.cm.tab10.colors  # pick a colormap
    color_map = {cochlea: colors[i % len(colors)] for i, cochlea in enumerate(cochleas)}
    if len(cochleas) == 1:
        color_map = {"L": colors[0], "R": colors[1]}

    # Track which cochlea names we have already added to the legend
    legend_added = set()

    trend_dict = {}

    for name_lr, grp in result.groupby("cochlea"):
        name, side = name_lr[:-1], name_lr[-1]
        if len(cochleas) == 1:
            label_name = name_lr
            color = color_map[side]
        else:
            label_name = name
            color = color_map[name]

        x_positions = grp["x_pos"] + offset_map[side]
        ax.scatter(
            x_positions,
            grp["value"],
            label=label_name if label_name not in legend_added else None,
            s=60,
            alpha=0.8,
            marker="o" if side == "L" else "x",
            color=color,
        )

        if name not in legend_added:
            legend_added.add(name)

        if trendlines:
            sorted_idx = np.argsort(x_positions)
            x_sorted = np.array(x_positions)[sorted_idx]
            y_sorted = np.array(grp["value"])[sorted_idx]
            trend_dict[name_lr] = {"x_sorted": x_sorted,
                                   "y_sorted": y_sorted,
                                   "side": side,
                                   }

    if trendlines:
        def get_trendline_values(trend_dict, side):
            x_sorted = [trend_dict[k]["x_sorted"] for k in trend_dict.keys() if trend_dict[k]["side"] == side][0]
            y_sorted_all = [trend_dict[k]["y_sorted"] for k in trend_dict.keys() if trend_dict[k]["side"] == side]
            y_sorted = []
            for num in range(len(x_sorted)):
                y_sorted.append(np.mean([y[num] for y in y_sorted_all]))
            return x_sorted, y_sorted

        # Trendline left
        x_sorted, y_sorted = get_trendline_values(trend_dict, "L")

        trend_l, = ax.plot(
            x_sorted,
            y_sorted,
            linestyle="dotted",
            color="grey",
            alpha=0.7
        )

        x_sorted, y_sorted = get_trendline_values(trend_dict, "R")
        trend_r, = ax.plot(
            x_sorted,
            y_sorted,
            linestyle="dashed",
            color="grey",
            alpha=0.7
        )
        trendline_legend = ax.legend(handles=[trend_l, trend_r], loc='lower center')
        trendline_legend = ax.legend(
            handles=[trend_l, trend_r],
            labels=["Injected", "Non-Injected"],
            loc="lower center",
            fontsize=legend_size,
            title="Trendlines"
        )
        # Add the legend manually to the Axes.
        ax.add_artist(trendline_legend)

    # Create combined tick positions & labels
    main_ticks = range(len(bin_labels))
    # add a final tick for label '>64k'
    ax.set_xticks([pos + offset_map["L"] for pos in main_ticks] +
                  [pos + offset_map["R"] for pos in main_ticks])
    ax.set_xticklabels(["I"] * len(main_ticks) + ["N"] * len(main_ticks), fontsize=sub_tick_label_size)

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

    ax.legend(title="Cochlea", fontsize=legend_size)

    plt.tight_layout()
    prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

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
    fig_04c(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04c.{FILE_EXTENSION}"),
            plot=args.plot, plot_by_side=True, use_alias=use_alias)

    # D: The transduction efficiency. We also plot GFP intensities.
    fig_04d(chreef_data,
            save_path=os.path.join(args.figure_dir,  f"fig_04d_transduction.{FILE_EXTENSION}"),
            plot=args.plot, plot_by_side=True, use_alias=use_alias)
    # fig_04d(chreef_data,
    #         save_path=os.path.join(args.figure_dir, f"fig_04d_intensity.{FILE_EXTENSION}"),
    #         plot=args.plot, plot_by_side=True, intensity=True, use_alias=use_alias)

    fig_04e(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04e_transduction.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, trendlines=True)
    # fig_04e(chreef_data,
    #         save_path=os.path.join(args.figure_dir, f"fig_04e_intensity.{FILE_EXTENSION}"),
    #         plot=args.plot, intensity=True, use_alias=use_alias)

    chreef_data_gerbil = get_chreef_data(animal="gerbil")
    fig_04d(chreef_data_gerbil,
            save_path=os.path.join(args.figure_dir, f"fig_04d_gerbil_transduction.{FILE_EXTENSION}"),
            plot=args.plot, plot_by_side=True, gerbil=True, use_alias=use_alias)

    # fig_04d(chreef_data_gerbil,
    #         save_path=os.path.join(args.figure_dir, f"fig_04d_gerbil_intensity.{FILE_EXTENSION}"),
    #         plot=args.plot, plot_by_side=True, intensity=True, use_alias=use_alias)

    fig_04e(chreef_data_gerbil,
            save_path=os.path.join(args.figure_dir, f"fig_04e_gerbil_transduction.{FILE_EXTENSION}"),
            plot=args.plot, gerbil=True, use_alias=use_alias)

    # fig_04e(chreef_data_gerbil,
    #         save_path=os.path.join(args.figure_dir, f"fig_04e_gerbil_intensity.{FILE_EXTENSION}"),
    #         plot=args.plot, intensity=True, use_alias=use_alias)


if __name__ == "__main__":
    main()
