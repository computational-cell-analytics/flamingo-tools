import json
import os
import sys
from glob import glob
from subprocess import run

import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import threshold_otsu

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target, get_s3_path
from flamingo_tools.measurements import compute_object_measures

sys.path.append("../figures")

# Map from cochlea names to channels
COCHLEAE_FOR_SUBTYPES = {
    "M_LR_000099_L": ["PV", "Calb1", "Lypd1"],
    "M_LR_000214_L": ["PV", "CR", "Calb1"],
    "M_AMD_N62_L": ["PV", "CR", "Calb1"],
    "M_AMD_N180_R": ["CR", "Ntng1", "CTBP2"],
    "M_AMD_N180_L": ["CR", "Ntng1", "Lypd1"],
    "M_LR_000184_R": ["PV", "Prph"],
    "M_LR_000184_L": ["PV", "Prph"],
    # Mutant / some stuff is weird.
    # "M_AMD_Runx1_L": ["PV", "Lypd1", "Calb1"],
    # This one still has to be stitched:
    # "M_LR_000184_R": {"PV", "Prph"},
}
REGULAR_COCHLEAE = [
    "M_LR_000099_L", "M_LR_000214_L", "M_AMD_N62_L", "M_LR_000184_R", "M_LR_000184_L"
]

# Map from channels to subtypes.
# Comment Aleyna:
# The signal will be a gradient between different subtypes:
# For example CR is expressed more, is brigther,
# in type 1a SGNs but exist in type Ib SGNs and to a lesser extent in type 1c.
# Same is also true for other markers so we will need to set a threshold for each.
# Luckily the signal seems less variable compared to GFP.
CHANNEL_TO_TYPE = {
    "CR": "Type-Ia",
    "Calb1": "Type-Ib",
    "Lypd1": "Type-Ic",
    "Prph": "Type-II",
    "Ntng1": "Type-Ib/c",
}

# For custom thresholds.
THRESHOLDS = {
    "M_LR_000214_L": {
    },
    "M_AMD_N62_L": {
    },
}

# For consistent colors.
ALL_COLORS = ["red", "blue", "orange", "yellow", "cyan", "magenta", "green", "purple"]
COLORS = {}

PLOT_OUT = "./subtype_plots"

# TODO: updates based on Aleyna's feedback.
# Subtype mapping

# Combined visualization for the cochleae
# Can we visualize the tonotopy in subtypes and not stainings?
# It would also be good to have subtype percentages per cochlea and pooled together as a diagram and tonotopy?
# This would help to see if different staining gives same/similar results.
# Type Ia ; CR+ / Calb1- or Calb1- / Lypd1-
# Type Ib: CR+ / Calb1+ or Calb1+ / Lypd1+
# Type Ic: CR-/Calb1+ - or Calb1- / Lypd1+
# Type II: CR-/Calb1- or Calb1- / Lypd1- or Prph+

# > It's good to see that for the N mice the Ntng1C and Lypd1 separate from CR so well on the thresholds. Can I visualize these samples ones segmentation masks are done to verify the Ntng1C thresholds? As this is a quite clear signal I'm not sure if taking the middle of the histogram would be the best choice.
# The segmentations are in MoBIE already. I need to send you the tables for analyzing the signals. Will send them later.

# > Where are we at PV-Prph segmentation results from MLR184_L and R for SGN type II analysis? This would hopefully give <5% Prph+ cells.
# The cochleae are in MoBIE. Segmentation and Prph signal look good! I will include it in the next analysis.
# Need tonotopic mapping from Martin and then compute the intensities.


def check_processing_status():
    s3 = create_s3_target()

    # For checking the dataset names.
    # content = s3.open(f"{BUCKET_NAME}/project.json", mode="r", encoding="utf-8")
    # info = json.loads(content.read())
    # datasets = info["datasets"]
    # for name in datasets:
    #     print(name)
    # breakpoint()

    missing_tables = {}

    for cochlea, channels in COCHLEAE_FOR_SUBTYPES.items():
        if cochlea not in REGULAR_COCHLEAE:
            continue
        try:
            content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        except FileNotFoundError:
            print(cochlea, "is not yet on MoBIE")
            print()
            continue
        info = json.loads(content.read())
        sources = info["sources"]

        channels_found = [name for name in channels if name in sources]
        channels_missing = [name for name in channels if name not in sources]

        print("Cochlea:", cochlea)
        print("Found the expected channels:", channels_found)
        if channels_missing:
            print("Missing the expected channels:", channels_missing)

        if "SGN_v2" in sources:
            print("SGN segmentation is present with name SGN_v2")
            seg_name = "SGN-v2"
            table_folder = "tables/SGN_v2"
        elif "PV_SGN_v2" in sources:
            print("SGN segmentation is present with name PV_SGN_v2")
            seg_name = "PV-SGN-v2"
            table_folder = "tables/PV_SGN_v2"
        elif "CR_SGN_v2" in sources:
            print("SGN segmentation is present with name CR_SGN_v2")
            seg_name = "CR-SGN-v2"
            table_folder = "tables/CR_SGN_v2"
        else:
            print("SGN segmentation is MISSING")
            print()
            continue

        # Check which tables we have.
        if cochlea == "M_AMD_N180_L":  # we need all intensity measures here
            seg_names = ["CR-SGN-v2", "Ntng1-SGN-v2", "Lypd1-SGN-v2"]
            expected_tables = [f"{chan}_{sname}_object-measures.tsv" for chan in channels for sname in seg_names]
        elif cochlea == "M_AMD_N180_R":
            seg_names = ["CR-SGN-v2", "Ntng1-SGN-v2"]
            expected_tables = [f"{chan}_{sname}_object-measures.tsv" for chan in channels for sname in seg_names]
        else:
            expected_tables = [f"{chan}_{seg_name}_object-measures.tsv" for chan in channels]

        tables = s3.ls(os.path.join(BUCKET_NAME, cochlea, table_folder))
        tables = [os.path.basename(tab) for tab in tables]

        this_missing_tables = []
        for exp_tab in expected_tables:
            if exp_tab not in tables:
                print("Missing table:", exp_tab)
                this_missing_tables.append(exp_tab)
        missing_tables[cochlea] = this_missing_tables
        print()

    return missing_tables


def require_missing_tables(missing_tables):
    output_root = "./object_measurements"

    for cochlea, missing_tabs in missing_tables.items():
        if cochlea not in REGULAR_COCHLEAE:
            continue
        for missing in missing_tabs:
            channel = missing.split("_")[0]
            seg_name = missing.split("_")[1].replace("-", "_")
            print("Computing intensities for cochlea:", cochlea, "segmentation:", seg_name, "channel:", channel)

            img_s3 = f"{cochlea}/images/ome-zarr/{channel}.ome.zarr"
            seg_s3 = f"{cochlea}/images/ome-zarr/{seg_name}.ome.zarr"
            seg_table_s3 = f"{cochlea}/tables/{seg_name}/default.tsv"
            img_path, _ = get_s3_path(img_s3)
            seg_path, _ = get_s3_path(seg_s3)

            output_folder = os.path.join(output_root, cochlea)
            os.makedirs(output_folder, exist_ok=True)
            output_table_path = os.path.join(
                output_folder, f"{channel}_{seg_name.replace('_', '-')}_object-measures.tsv"
            )
            compute_object_measures(
                image_path=img_path,
                segmentation_path=seg_path,
                segmentation_table_path=seg_table_s3,
                output_table_path=output_table_path,
                image_key="s0",
                segmentation_key="s0",
                s3_flag=True,
                component_list=[1],
                n_threads=16,
            )

            # S3 upload
            run(["rclone", "--progress", "copyto", output_folder,
                 f"cochlea-lightsheet:cochlea-lightsheet/{cochlea}/tables/{seg_name}"])


def compile_data_for_subtype_analysis():
    s3 = create_s3_target()

    output_folder = "./subtype_analysis"
    os.makedirs(output_folder, exist_ok=True)

    for cochlea, channels in COCHLEAE_FOR_SUBTYPES.items():
        if cochlea not in REGULAR_COCHLEAE:
            continue
        if "PV" in channels:
            reference_channel = "PV"
            seg_name = "PV_SGN_v2"
        else:
            assert "CR" in channels
            reference_channel = "CR"
            seg_name = "CR_SGN_v2"
        reference_channel, seg_name

        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the segmentation table.
        seg_source = sources[seg_name]
        table_folder = os.path.join(
            BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
        )
        table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        # Get the SGNs in the main component
        table = table[table.component_labels == 1]
        valid_sgns = table.label_id

        output_table = {"label_id": table.label_id.values, "frequency[kHz]": table["frequency[kHz]"]}

        # Analyze the different channels (= different subtypes).
        reference_intensity = None
        for channel in channels:
            # Load the intensity table.
            intensity_path = os.path.join(table_folder, f"{channel}_{seg_name.replace('_', '-')}_object-measures.tsv")
            table_content = s3.open(intensity_path, mode="rb")

            intensities = pd.read_csv(table_content, sep="\t")
            intensities = intensities[intensities.label_id.isin(valid_sgns)]
            assert len(table) == len(intensities)
            assert (intensities.label_id.values == table.label_id.values).all()

            medians = intensities["median"].values
            output_table[f"{channel}_median"] = medians
            if channel == reference_channel:
                reference_intensity = medians
            else:
                assert reference_intensity is not None
                output_table[f"{channel}_ratio_{reference_channel}"] = medians / reference_intensity

        out_path = os.path.join(output_folder, f"{cochlea}_subtype_analysis.tsv")
        output_table = pd.DataFrame(output_table)
        output_table.to_csv(out_path, sep="\t", index=False)


def _plot_histogram(table, column, name, show_plots, class_names=None, apply_threshold=True):
    data = table[column].values
    threshold = threshold_otsu(data)

    fig, ax = plt.subplots(1)
    ax.hist(data, bins=24)
    if apply_threshold:
        ax.axvline(x=threshold, color='red', linestyle='--')
        ax.set_title(f"{name}\n threshold: {threshold}")
    else:
        ax.set_title(name)

    if show_plots:
        plt.show()
    else:
        os.makedirs(PLOT_OUT, exist_ok=True)
        plt.savefig(f"{PLOT_OUT}/{name}.png")

    if class_names is not None:
        assert len(class_names) == 2
        c0, c1 = class_names
        subtype_classification = [c0 if datum < threshold else c1 for datum in data]
        return subtype_classification


def _plot_2d(ratios, name, show_plots, classification=None, colors=None):
    fig, ax = plt.subplots(1)
    assert len(ratios) == 2
    keys = list(ratios.keys())
    k1, k2 = keys

    if classification is None:
        ax.scatter(ratios[k1, k2])

    else:
        assert colors is not None
        unique_labels = set(classification)
        for lbl in unique_labels:
            mask = [ll == lbl for ll in classification]
            ax.scatter(
                [ratios[k1][i] for i in range(len(classification)) if mask[i]],
                [ratios[k2][i] for i in range(len(classification)) if mask[i]],
                c=colors[lbl], label=lbl
            )

        ax.legend()

    ax.set_xlabel(k1)
    ax.set_ylabel(k2)
    ax.set_title(name)

    if show_plots:
        plt.show()
    else:
        os.makedirs(PLOT_OUT, exist_ok=True)
        plt.savefig(f"{PLOT_OUT}/{name}.png")


def _plot_tonotopic_mapping(freq, classification, name, colors, show_plots):
    from util import frequency_mapping

    frequency_mapped = frequency_mapping(freq, classification, categorical=True)
    result = next(iter(frequency_mapped.values()))
    bin_labels = pd.unique(result["octave_band"])
    band_to_x = {band: i for i, band in enumerate(bin_labels)}
    x_positions = result["octave_band"].map(band_to_x)

    fig, ax = plt.subplots(figsize=(8, 4))
    for cat, vals in frequency_mapped.items():
        ax.scatter(x_positions, vals.value, label=cat, color=colors[cat])
    ax.legend()
    ax.set_title(name)

    if show_plots:
        plt.show()
    else:
        os.makedirs(PLOT_OUT, exist_ok=True)
        plt.savefig(f"{PLOT_OUT}/{name}.png")
    plt.close()


def analyze_subtype_data_regular(show_plots=True):
    global PLOT_OUT, COLORS  # noqa
    PLOT_OUT = "subtype_plots/regular_mice"

    files = sorted(glob("./subtype_analysis/*.tsv"))

    for ff in files:
        cochlea = os.path.basename(ff)[:-len("_subtype_analysis.tsv")]
        if cochlea not in REGULAR_COCHLEAE:
            continue
        print(cochlea)
        channels = COCHLEAE_FOR_SUBTYPES[cochlea]

        reference_channel = "PV"
        assert channels[0] == reference_channel

        tab = pd.read_csv(ff, sep="\t")

        # 1.) Plot simple intensity histograms, including otsu threshold.
        # for chan in channels:
        #     column = f"{chan}_median"
        #     name = f"{cochlea}_{chan}_histogram"
        #     _plot_histogram(tab, column, name, show_plots, apply_threshold=chan != reference_channel)

        # 2.) Plot ratio histograms, including otsu threshold.
        ratios = {}
        classification = []
        for chan in channels[1:]:
            column = f"{chan}_ratio_{reference_channel}"
            name = f"{cochlea}_{chan}_histogram_ratio_{reference_channel}"
            chan_classification = _plot_histogram(
                tab, column, name, class_names=[f"{chan}-", f"{chan}+"], show_plots=show_plots
            )
            classification.append(chan_classification)
            ratios[f"{chan}_{reference_channel}"] = tab[column].values

        # Unify the classification and assign colors
        cls1, cls2 = classification[0], classification[1]
        assert len(cls1) == len(cls2)
        classification = [f"{c1} / {c2}" for c1, c2 in zip(cls1, cls2)]

        unique_labels = set(classification)
        for label in unique_labels:
            if label in COLORS:
                continue
            if COLORS:
                last_color = list(COLORS.values())[-1]
                next_color = ALL_COLORS[ALL_COLORS.index(last_color) + 1]
                COLORS[label] = next_color
            else:
                COLORS[label] = ALL_COLORS[0]

        # 3.) Plot tonotopic mapping.
        freq = tab["frequency[kHz]"].values
        assert len(freq) == len(classification)
        name = f"{cochlea}_tonotopic_mapping"
        _plot_tonotopic_mapping(freq, classification, name=name, colors=COLORS, show_plots=show_plots)

        # 4.) Plot 2D space of ratios.
        name = f"{cochlea}_2d"
        _plot_2d(ratios, name, show_plots, classification=classification, colors=COLORS)


# General notes:
# See:
def main():
    missing_tables = check_processing_status()
    require_missing_tables(missing_tables)
    compile_data_for_subtype_analysis()

    # analyze_subtype_data_regular(show_plots=False)

    # TODO
    # analyze_subtype_data_N_mice()

    # CTBP2 stain
    # analyze_subtype_data_syn_mice()


if __name__ == "__main__":
    main()
