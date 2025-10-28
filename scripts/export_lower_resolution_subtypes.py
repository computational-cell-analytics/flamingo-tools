import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
# from skimage.segmentation import relabel_sequential

COCHLEA_DICT = {
    "M_LR_000099_L": {"seg_data": "PV_SGN_v2", "subtype": ["Calb1", "Lypd1"]},
    "M_LR_000184_L": {"seg_data": "SGN_v2b", "subtype": ["Prph"]},
    "M_LR_000184_R": {"seg_data": "SGN_v2b", "subtype": ["Prph"]},
    "M_LR_000260_L": {"seg_data": "SGN_v2", "subtype": ["Prph", "Tuj1"]},
}


STAIN_TO_TYPE = {
    # Combinations of Calb1 and CR:
    "CR+/Calb1+": "Type Ib",
    "CR-/Calb1+": "Type IbIc",  # Calb1 is expressed at Ic less than Lypd1 but more then CR
    "CR+/Calb1-": "Type Ia",
    "CR-/Calb1-": "Type II",

    # Combinations of Calb1 and Lypd1:
    "Calb1+/Lypd1+": "Type IbIc",
    "Calb1+/Lypd1-": "Type Ib",
    "Calb1-/Lypd1+": "Type Ic",
    "Calb1-/Lypd1-": "inconclusive",  # Can be Type Ia or Type II

    # Combinations of Prph and Tuj1:
    "Prph+/Tuj1+": "Type II",
    "Prph+/Tuj1-": "Type II",
    "Prph-/Tuj1+": "Type I",
    "Prph-/Tuj1-": "inconclusive",

    # Prph is isolated.
    "Prph+": "Type II",
    "Prph-": "Type I",
}


def types_for_stain(stains):
    stains.sort()
    assert len(stains) in (1, 2)
    if len(stains) == 1:
        combinations = [f"{stains[0]}+", f"{stains[0]}-"]
    else:
        combinations = [
            f"{stains[0]}+/{stains[1]}+",
            f"{stains[0]}+/{stains[1]}-",
            f"{stains[0]}-/{stains[1]}+",
            f"{stains[0]}-/{stains[1]}-"
        ]
    types = list(set([STAIN_TO_TYPE[stain] for stain in combinations]))
    return types


def stain_expression_from_subtype(subtype, stains):
    assert len(stains) in (1, 2)
    dic_list = []
    if len(stains) == 1:
        possible_key = [
            key for key in STAIN_TO_TYPE.keys()
            if STAIN_TO_TYPE[key] == subtype and len(key.split("/")) != 2 and stains[0] in key
        ][0]
        dic = {stains[0]: possible_key[-1:]}
        dic_list.append(dic)

    else:
        possible_keys = [
            key for key in STAIN_TO_TYPE.keys()
            if STAIN_TO_TYPE[key] == subtype and len(key.split("/")) > 1 and all([stain in key for stain in stains])
        ]
        for key in possible_keys:
            stain1 = key.split("/")[0][:-1]
            stain2 = key.split("/")[1][:-1]
            expression1 = key.split("/")[0][-1:]
            expression2 = key.split("/")[1][-1:]
            dic = {stain1: expression1, stain2: expression2}
            dic_list.append(dic)

    return dic_list


def stain_to_type(stain):
    # Normalize the staining string.
    stains = stain.replace(" ", "").split("/")
    assert len(stains) in (1, 2)

    if len(stains) == 1:
        stain_norm = stain
    else:
        s1, s2 = sorted(stains)
        stain_norm = f"{s1}/{s2}"

    if stain_norm not in STAIN_TO_TYPE:
        breakpoint()
        raise ValueError(f"Invalid stain combination: {stain_norm}")

    return STAIN_TO_TYPE[stain_norm], stain_norm


def filter_subtypes(cochlea, segmentation, seg_name, subtype):
    """Filter segmentation with marker labels.
    Positive segmentation instances are set to 1, negative to 2.
    """
    internal_path = os.path.join(cochlea, "tables",  seg_name, "default.tsv")
    tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    with fs.open(tsv_path, "r") as f:
        table_seg = pd.read_csv(f, sep="\t")

    # get stains
    stains = [column.split("_")[1] for column in list(table_seg.columns) if "marker_" in column]
    stains.sort()

    stain_dict = stain_expression_from_subtype(subtype, stains)
    if len(stain_dict) == 0:
        raise ValueError("The dictionary containing stain information must have at least one entry. Check parameters.")

    subset = table_seg.copy()

    for dic in stain_dict:
        for stain in dic.keys():
            expression_value = 1 if dic[stain] == "+" else 2
            subset = subset.loc[subset[f"marker_{stain}"] == expression_value]

    label_ids_subtype = list(subset["label_id"])
    print(f"subtype {subtype} with {len(label_ids_subtype)} instances")

    filter_mask = ~np.isin(segmentation, label_ids_subtype)
    segmentation[filter_mask] = 0
    filter_mask = np.isin(segmentation, label_ids_subtype)
    segmentation[filter_mask] = 1

    segmentation = segmentation.astype("uint16")
    return segmentation


def export_lower_resolution(args):

    cochlea = args.cochlea
    subtype_stains = args.stains
    # iterate through exporting lower resolutions
    for scale in args.scale:
        output_folder = os.path.join(args.output_folder, cochlea, f"scale{scale}")
        os.makedirs(output_folder, exist_ok=True)
        if cochlea in COCHLEA_DICT.keys():
            if subtype_stains is None:
                subtype_stains = COCHLEA_DICT[cochlea]["subtype"]
            seg_name = COCHLEA_DICT[cochlea]["seg_data"]
        else:
            raise ValueError(f"Cochlea {cochlea} is not in the dictionary. Check values.")

        print(f"Subtype stains: {subtype_stains}.")
        subtypes = types_for_stain(subtype_stains)
        subtypes.sort()

        for subtype in subtypes:

            input_key = f"s{scale}"
            internal_path = os.path.join(cochlea, "images",  "ome-zarr", f"{seg_name}.ome.zarr")
            s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
            with zarr.open(s3_store, mode="r") as f:
                data = f[input_key][:]

            out_path = os.path.join(output_folder, f"{seg_name}_{subtype.replace(" ", "")}.tif")
            if os.path.exists(out_path):
                continue

            print("Data shape", data.shape)

            print(f"Filtering subtype: {subtype}.")
            data = filter_subtypes(cochlea, data, seg_name=seg_name, subtype=subtype)
            tifffile.imwrite(out_path, data, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", nargs="+", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--stains", nargs="+", type=str, default=None)
    args = parser.parse_args()

    export_lower_resolution(args)


if __name__ == "__main__":
    main()
