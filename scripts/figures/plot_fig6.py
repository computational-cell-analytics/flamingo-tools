import argparse
import json
import numpy as np
import os
import pickle

import matplotlib.ticker as mticker
import pandas as pd
import matplotlib.pyplot as plt

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target
from util import literature_reference_values_gerbil, prism_cleanup_axes, prism_style, SYNAPSE_DIR_ROOT
from util import frequency_mapping, custom_formatter_1, export_legend

FILE_EXTENSION = "png"
png_dpi = 300


INTENSITY_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/LaVision-OTOF"  # noqa

# The cochlea for the CHReef analysis.
COCHLEAE_DICT = {
    "LaVision-OTOF23R": {"alias": "M02", "component": [4, 18, 7]},
    "LaVision-OTOF25R": {"alias": "M03", "component": [1]},
}

COLOR_LEFT = "#8E00DB"
COLOR_RIGHT = "#DB0063"
MARKER_LEFT = "o"
MARKER_RIGHT = "^"


def get_otof_data():
    s3 = create_s3_target()
    source_name = "IHC_LOWRES-v3"

    cache_path = "./otof_data.pkl"
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
        print(table.columns)

        # May need to be adjusted for some cochleae.
        component_labels = COCHLEAE_DICT[cochlea]["component"]
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        # The relevant values for analysis.
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "expression_classification"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skippped.")
            continue

        fname = f"{cochlea.replace('_', '-')}_rbOtof_IHC-LOWRES-v3_object-measures.tsv"
        intensity_file = os.path.join(INTENSITY_ROOT, fname)
        assert os.path.exists(intensity_file), intensity_file
        intensity_table = pd.read_csv(intensity_file, sep="\t")
        values = values.merge(intensity_table, on="label_id")

        chreef_data[cochlea] = values

    with open(cache_path, "wb") as f:
        pickle.dump(chreef_data, f)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


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


def plot_legend_fig06e(figure_dir):
    color_dict = {
        "O1": "#9C5027",
        "O2": "#67279C",
    }
    save_path = os.path.join(figure_dir, f"fig_06e_legend.{FILE_EXTENSION}")
    marker = ["o" for _ in color_dict]
    label = list(color_dict.keys())
    color = [color_dict[key] for key in color_dict.keys()]

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(m, c) for (c, m) in zip(color, marker)]
    legend = plt.legend(handles, label, loc=3, ncol=2, framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def _get_trendline_dict(trend_dict,):
    x_sorted = [trend_dict[k]["x_sorted"] for k in trend_dict.keys()]
    x_dict = {}
    for num in range(len(x_sorted[0])):
        x_dict[num] = {"pos": num, "values": []}

    for s in x_sorted:
        for num, pos in enumerate(s):
            x_dict[num]["values"].append(pos)

    y_sorted_all = [trend_dict[k]["y_sorted"] for k in trend_dict.keys()]
    y_dict = {}
    for num in range(len(x_sorted[0])):
        y_dict[num] = {"pos": num, "values": []}

    for num in range(len(x_sorted[0])):
        y_dict[num]["mean"] = np.mean([y[num] for y in y_sorted_all])
        y_dict[num]["stdv"] = np.std([y[num] for y in y_sorted_all])
    return x_dict, y_dict


def _get_trendline_params(trend_dict):
    x_dict, y_dict = _get_trendline_dict(trend_dict)

    x_values = []
    for key in x_dict.keys():
        x_values.append(min(x_dict[key]["values"]))
        x_values.append(max(x_dict[key]["values"]))

    y_values_center = []
    y_values_upper = []
    y_values_lower = []
    for key in y_dict.keys():
        y_values_center.append(y_dict[key]["mean"])
        y_values_center.append(y_dict[key]["mean"])

        y_values_upper.append(y_dict[key]["mean"] + y_dict[key]["stdv"])
        y_values_upper.append(y_dict[key]["mean"] + y_dict[key]["stdv"])

        y_values_lower.append(y_dict[key]["mean"] - y_dict[key]["stdv"])
        y_values_lower.append(y_dict[key]["mean"] - y_dict[key]["stdv"])

    return x_values, y_values_center, y_values_upper, y_values_lower


def fig_06e_octave(otof_data, save_path, plot=False, use_alias=True, trendline=False):
    prism_style()
    label_size = 20

    result = {"cochlea": [], "octave_band": [], "value": []}
    for name, values in otof_data.items():
        if use_alias:
            alias = COCHLEAE_DICT[name]["alias"]
        else:
            alias = name.replace("_", "").replace("0", "")

        freq = values["frequency[kHz]"].values
        marker_labels = values["expression_classification"].values
        octave_binned = frequency_mapping(freq, marker_labels, animal="mouse", transduction_efficiency=True)

        result["cochlea"].extend([alias] * len(octave_binned))
        result["octave_band"].extend(octave_binned.axes[0].values.tolist())
        result["value"].extend(octave_binned.values.tolist())

    result = pd.DataFrame(result)
    bin_labels = pd.unique(result["octave_band"])
    band_to_x = {band: i for i, band in enumerate(bin_labels)}
    result["x_pos"] = result["octave_band"].map(band_to_x)

    colors = {
        "M02": "#9C5027",
        "M03": "#67279C",
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    offset = 0.08
    trend_dict = {}
    for num, (name, grp) in enumerate(result.groupby("cochlea")):
        x_sorted = grp["x_pos"]
        x_positions = [x - len(grp["x_pos"]) // 2 * offset + offset * num for x in grp["x_pos"]]
        ax.scatter(x_positions, grp["value"], marker="o", label=name, s=80, alpha=1, color=colors[name])

        # y_values.append(list(grp["value"]))

        if trendline:
            sorted_idx = np.argsort(x_positions)
            x_sorted = np.array(x_positions)[sorted_idx]
            y_sorted = np.array(grp["value"])[sorted_idx]
            trend_dict[name] = {"x_sorted": x_sorted,
                                "y_sorted": y_sorted,
                                }
    # central line
    if trendline:
        #mean, std = _get_trendline_params(y_values)
        x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = _get_trendline_params(trend_dict)
        trend_center, = ax.plot(
            x_sorted,
            y_sorted,
            linestyle="dotted",
            color="gray",
            alpha=0.6,
            linewidth=3,
            zorder=2
        )
        # y_sorted_upper = np.array(mean) + np.array(std)
        # y_sorted_lower = np.array(mean) - np.array(std)
        # upper and lower standard deviation
        trend_upper, = ax.plot(
            x_sorted,
            y_sorted_upper,
            linestyle="solid",
            color="gray",
            alpha=0.08,
            zorder=0
        )
        trend_lower, = ax.plot(
            x_sorted,
            y_sorted_lower,
            linestyle="solid",
            color="gray",
            alpha=0.08,
            zorder=0
        )
        plt.fill_between(x_sorted, y_sorted_lower, y_sorted_upper,
                         color="gray", alpha=0.05, interpolate=True)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Octave band [kHz]", fontsize=label_size)

    ax.set_ylabel("Expression efficiency")
    # plt.legend(title="Cochlea")
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


def fig_06b(save_path, plot=False):
    """Box plot showing the counts for SGN and IHC per gerbil cochlea in comparison to literature values.
    """
    main_tick_size = 20
    main_label_size = 20
    prism_style()

    rows = 1
    columns = 3

    fig, ax = plt.subplots(rows, columns, figsize=(columns*3, rows*4))

    sgn_values = [18541]
    ihc_values = [1180]

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
    ax[2].set_xticklabels(["Synapses per IHC"], fontsize=main_label_size)
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
    prism_cleanup_axes(ax)

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
    parser.add_argument("-f", "--figure_dir", type=str, help="Output directory for plots.", default="./panels")
    args = parser.parse_args()
    plot = False

    otof_data = get_otof_data()
    plot_legend_fig06e(args.figure_dir)
    fig_06e_octave(otof_data, save_path=os.path.join(args.figure_dir, f"fig_06e.{FILE_EXTENSION}"), plot=plot,
                   trendline=False, gr)

    fig_06b(save_path=os.path.join(args.figure_dir, f"fig_06b.{FILE_EXTENSION}"), plot=plot)
    fig_06d(save_path=os.path.join(args.figure_dir, f"fig_06d.{FILE_EXTENSION}"), plot=plot)


if __name__ == "__main__":
    main()
