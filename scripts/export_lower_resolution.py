import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
# from skimage.segmentation import relabel_sequential


def filter_component(fs, segmentation, cochlea, seg_name, components):
    # First, we download the MoBIE table for this segmentation.
    internal_path = os.path.join(BUCKET_NAME, cochlea, "tables",  seg_name, "default.tsv")
    with fs.open(internal_path, "r") as f:
        table = pd.read_csv(f, sep="\t")

    # Then we get the ids for the components and us them to filter the segmentation.
    component_mask = np.isin(table.component_labels.values, components)
    keep_label_ids = table.label_id.values[component_mask].astype("int64")
    filter_mask = ~np.isin(segmentation, keep_label_ids)
    segmentation[filter_mask] = 0

    # segmentation, _, _ = relabel_sequential(segmentation)
    return segmentation


def export_lower_resolution(args):
    output_folder = os.path.join(args.output_folder, args.cochlea, f"scale{args.scale}")
    os.makedirs(output_folder, exist_ok=True)

    input_key = f"s{args.scale}"
    for channel in args.channels:
        out_path = os.path.join(output_folder, f"{channel}.tif")
        if os.path.exists(out_path):
            continue

        print("Exporting channel", channel)
        internal_path = os.path.join(args.cochlea, "images",  "ome-zarr", f"{channel}.ome.zarr")
        s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
        with zarr.open(s3_store, mode="r") as f:
            data = f[input_key][:]
        print(data.shape)
        if args.filter_by_components is not None:
            data = filter_component(fs, data, args.cochlea, channel, args.filter_by_components)
        if args.binarize:
            data = (data > 0).astype("uint16")
        tifffile.imwrite(out_path, data, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--channels", nargs="+", default=["PV", "VGlut3", "CTBP2"])
    parser.add_argument("--filter_by_components", nargs="+", type=int, default=None)
    parser.add_argument("--binarize", action="store_true")
    args = parser.parse_args()

    export_lower_resolution(args)


if __name__ == "__main__":
    main()
