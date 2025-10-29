import argparse
import json
import os
from typing import List, Optional

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.segmentation.chreef_utils import localize_median_intensities, find_annotations

MARKER_DIR_SUBTYPE = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes"
# The cochlea for the CHReef analysis.

COCHLEAE = {
    "M_AMD_N180_L": {"seg_data": "SGN_merged", "subtype": ["CR", "Lypd1", "Ntng1"], "intensity": "absolute"},
    "M_AMD_N180_R": {"seg_data": "SGN_merged", "subtype": ["CR", "Ntng1"], "intensity": "absolute"},
    "M_LR_000099_L": {"seg_data": "PV_SGN_v2", "subtype": ["Calb1", "Lypd1"], "intensity": "ratio"},
    "M_LR_000184_L": {"seg_data": "SGN_v2", "subtype": ["Prph"], "output_seg": "SGN_v2b", "intensity": "ratio"},
    "M_LR_000184_R": {"seg_data": "SGN_v2", "subtype": ["Prph"], "output_seg": "SGN_v2b", "intensity": "ratio"},
    "M_LR_000214_L": {"seg_data": "PV_SGN_v2", "subtype": ["Calb1"], "intensity": "ratio"},
    "M_LR_000260_L": {"seg_data": "SGN_v2", "subtype": ["Prph", "Tuj1"], "intensity": "ratio"},

}


def get_length_fraction_from_center(table, center_str):
    """Get 'length_fraction' parameter for center coordinate by averaging nearby segmentation instances.
    """
    center_coord = tuple([int(c) for c in center_str.split("-")])
    (cx, cy, cz) = center_coord
    offset = 20
    subset = table[
        (cx - offset < table["anchor_x"]) &
        (table["anchor_x"] < cx + offset) &
        (cy - offset < table["anchor_y"]) &
        (table["anchor_y"] < cy + offset) &
        (cz - offset < table["anchor_z"]) &
        (table["anchor_z"] < cz + offset)
    ]
    length_fraction = list(subset["length_fraction"])
    length_fraction = float(sum(length_fraction) / len(length_fraction))
    return length_fraction


def apply_nearest_threshold(intensity_dic, table_seg, table_measurement, column="median", suffix="labels"):
    """Apply threshold to nearest segmentation instances.
    Crop centers are transformed into the "length fraction" parameter of the segmentation table.
    This avoids issues with the spiral shape of the cochlea and maps the assignment onto the Rosenthal"s canal.
    """
    # assign crop centers to length fraction of Rosenthal"s canal
    lf_intensity = {}
    for key in intensity_dic.keys():
        length_fraction = get_length_fraction_from_center(table_seg, key)
        intensity_dic[key]["length_fraction"] = length_fraction
        lf_intensity[length_fraction] = {"threshold": intensity_dic[key]["median_intensity"]}

    # get limits for checking marker thresholds
    lf_intensity = dict(sorted(lf_intensity.items()))
    lf_fractions = list(lf_intensity.keys())
    # start of cochlea
    lf_limits = [0]
    # half distance between block centers
    for i in range(len(lf_fractions) - 1):
        lf_limits.append((lf_fractions[i] + lf_fractions[i+1]) / 2)
    # end of cochlea
    lf_limits.append(1)

    marker_labels = [0 for _ in range(len(table_seg))]
    table_seg.loc[:, f"marker_{suffix}"] = marker_labels
    for num, fraction in enumerate(lf_fractions):
        subset_seg = table_seg[
            (table_seg["length_fraction"] > lf_limits[num]) &
            (table_seg["length_fraction"] < lf_limits[num + 1])
        ]
        # assign values based on limits
        threshold = lf_intensity[fraction]["threshold"]
        label_ids_seg = subset_seg["label_id"]

        subset_measurement = table_measurement[table_measurement["label_id"].isin(label_ids_seg)]
        subset_positive = subset_measurement[subset_measurement[column] >= threshold]
        subset_negative = subset_measurement[subset_measurement[column] < threshold]
        label_ids_pos = list(subset_positive["label_id"])
        label_ids_neg = list(subset_negative["label_id"])

        table_seg.loc[table_seg["label_id"].isin(label_ids_pos), f"marker_{suffix}"] = 1
        table_seg.loc[table_seg["label_id"].isin(label_ids_neg), f"marker_{suffix}"] = 2

    return table_seg


def find_thresholds(cochlea_annotations, cochlea, data_seg, table_measurement, column="median", pattern=None):
    # Find the median intensities by averaging the individual annotations for specific crops
    annotation_dics = {}
    annotated_centers = []
    for annotation_dir in cochlea_annotations:
        print(f"Localizing threshold with median intensities for {os.path.basename(annotation_dir)}.")
        annotation_dic = localize_median_intensities(annotation_dir, cochlea, data_seg,
                                                     table_measurement, column=column, pattern=pattern)
        annotated_centers.extend(annotation_dic["center_strings"])
        annotation_dics[annotation_dir] = annotation_dic

    annotated_centers = list(set(annotated_centers))
    intensity_dic = {}
    # loop over all annotated blocks
    for annotated_center in annotated_centers:
        intensities = []
        annotator_success = []
        annotator_failure = []
        annotator_missing = []
        # loop over annotated block from single user
        for annotator_key in annotation_dics.keys():
            if annotated_center not in annotation_dics[annotator_key]["center_strings"]:
                annotator_missing.append(os.path.basename(annotator_key))
                continue
            else:
                median_intensity = annotation_dics[annotator_key][annotated_center]["median_intensity"]
                if median_intensity is None:
                    print(f"No threshold for {os.path.basename(annotator_key)} and crop {annotated_center}.")
                    annotator_failure.append(os.path.basename(annotator_key))
                else:
                    intensities.append(median_intensity)
                    annotator_success.append(os.path.basename(annotator_key))

        if len(intensities) == 0:
            print(f"No viable annotation for cochlea {cochlea} and crop {annotated_center}.")
            median_int_avg = None
        else:
            median_int_avg = float(sum(intensities) / len(intensities)),

        intensity_dic[annotated_center] = {
            "median_intensity": median_int_avg,
            "annotation_success": annotator_success,
            "annotation_failure": annotator_failure,
            "annotation_missing": annotator_missing,
        }

    return intensity_dic


def evaluate_marker_annotation(
    cochleae: List[str],
    output_dir: str,
    annotation_dirs: Optional[List[str]] = None,
    seg_name: str = "SGN_v2",
    marker_name: str = "Calb1",
    threshold_save_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """Evaluate marker annotations of a single or multiple annotators.
    Segmentation instances are assigned a positive (1) or negative label (2)
    in form of the "marker_label" component of the output segmentation table.
    The assignment is based on the median intensity supplied by a measurement table.
    Instances not considered for the assignment are labeled as 0.

    Args:
        cochleae: List of cochlea
        output_dir: Output directory for segmentation table with "marker_label" in format <cochlea>_<marker>_<seg>.tsv
        annotation_dirs: List of directories containing marker annotations by annotator(s).
        seg_name: Identifier for segmentation.
        marker_name: Identifier for marker stain.
        threshold_save_dir: Optional directory for saving the thresholds.
        force: Whether to overwrite already existing results.
    """
    input_key = "s0"

    if annotation_dirs is None:
        marker_dir = MARKER_DIR_SUBTYPE
        annotation_dirs = [entry.path for entry in os.scandir(marker_dir)
                           if os.path.isdir(entry) and "Result" in entry.name]

    for cochlea in cochleae:
        data_name = COCHLEAE[cochlea]["seg_data"]
        if "output_seg" in list(COCHLEAE[cochlea].keys()):
            output_seg = COCHLEAE[cochlea]["output_seg"]
        else:
            output_seg = data_name

        seg_string = "-".join(output_seg.split("_"))
        cochlea_str = "-".join(cochlea.split("_"))
        subtypes = COCHLEAE[cochlea]["subtype"]
        subtype_str = "_".join(subtypes)
        out_path = os.path.join(output_dir, f"{cochlea_str}_{subtype_str}_{seg_string}.tsv")
        if os.path.exists(out_path) and not force:
            continue

        # Get the segmentation data and table.
        input_path = f"{cochlea}/images/ome-zarr/{data_name}.ome.zarr"
        input_path, fs = get_s3_path(input_path)
        data_seg = read_image_data(input_path, input_key)

        table_seg_path = f"{cochlea}/tables/{output_seg}/default.tsv"
        table_path_s3, fs = get_s3_path(table_seg_path)
        with fs.open(table_path_s3, "r") as f:
            table_seg = pd.read_csv(f, sep="\t")

        # Check whether to use intensity ratio of subtype / PV or object measures for thresholding
        intensity_mode = COCHLEAE[cochlea]["intensity"]

        # iterate through subtypes
        for subtype in subtypes:
            pattern = subtype
            if intensity_mode == "ratio":
                table_measurement_path = f"{cochlea}/tables/{data_name}/subtype_ratio.tsv"
                column = f"{subtype}_ratio_PV"
            elif intensity_mode == "absolute":
                table_measurement_path = f"{cochlea}/tables/{data_name}/{subtype}_{seg_string}_object-measures.tsv"
                column = "median"
            else:
                raise ValueError("Choose either 'ratio' or 'median' as intensity mode.")

            table_path_s3, fs = get_s3_path(table_measurement_path)
            with fs.open(table_path_s3, "r") as f:
                table_measurement = pd.read_csv(f, sep="\t")

            cochlea_annotations = [a for a in annotation_dirs
                                   if len(find_annotations(a, cochlea, subtype)["center_strings"]) != 0]
            print(f"Evaluating data for cochlea {cochlea} in {cochlea_annotations}.")

            # Find the thresholds from the annotated blocks and save them if specified.
            intensity_dic = find_thresholds(cochlea_annotations, cochlea, data_seg,
                                            table_measurement, column=column, pattern=pattern)
            if threshold_save_dir is not None:
                os.makedirs(threshold_save_dir, exist_ok=True)
                threshold_out_path = os.path.join(threshold_save_dir, f"{cochlea_str}_{subtype}_{seg_string}.json")
                with open(threshold_out_path, "w") as f:
                    json.dump(intensity_dic, f, sort_keys=True, indent=4)

            # load measurement table of output segmentation
            if "output_seg" in list(COCHLEAE[cochlea].keys()):
                output_seg = COCHLEAE[cochlea]["output_seg"]
                table_measurement_path = f"{cochlea}/tables/{output_seg}/subtype_ratio.tsv"
                table_path_s3, fs = get_s3_path(table_measurement_path)
                with fs.open(table_path_s3, "r") as f:
                    table_measurement = pd.read_csv(f, sep="\t")

            # Apply the threshold to all SGNs.
            table_seg = apply_nearest_threshold(
                intensity_dic, table_seg, table_measurement, column=column, suffix=subtype,
            )

        # Save the table with positives / negatives for all SGNs.
        os.makedirs(output_dir, exist_ok=True)
        table_seg.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Assign each segmentation instance a marker based on annotation thresholds."
    )

    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-a", "--annotation_dirs", type=str, nargs="+", default=None,
                        help="Directories containing marker annotations.")
    parser.add_argument("--threshold_save_dir", "-t")
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args()
    evaluate_marker_annotation(
        args.cochlea, args.output, args.annotation_dirs, threshold_save_dir=args.threshold_save_dir, force=args.force,
    )


if __name__ == "__main__":
    main()
