import argparse
import json
import os

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, create_s3_target, get_s3_path
from skimage.morphology import ball
from tqdm import tqdm


# TODO
def export_synapse_detections(cochlea, scale, output_folder, synapse_name, reference_ihcs, max_dist, radius):
    s3 = create_s3_target()

    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    sources = info["sources"]

    # Load the synapse table.
    syn = sources[synapse_name]["spots"]
    rel_path = syn["tableData"]["tsv"]["relativePath"]
    table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")

    syn_table = pd.read_csv(table_content, sep="\t")
    syn_table = syn_table[syn_table.distance_to_ihc <= max_dist]

    # Get the reference segmentation info.
    reference_seg_info = sources[reference_ihcs]["segmentation"]

    # Get the segmentation table.
    rel_path = reference_seg_info["tableData"]["tsv"]["relativePath"]
    seg_table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
    seg_table = pd.read_csv(seg_table_content, sep="\t")

    # Only keep synapses that match to segmented IHCs of the main component.
    valid_ihcs = seg_table[seg_table.component_labels == 1].label_id
    syn_table = syn_table[syn_table.matched_ihc.isin(valid_ihcs)]

    # Get the reference shape at the given scale level.
    seg_path = os.path.join(cochlea, reference_seg_info["imageData"]["ome.zarr"]["relativePath"])
    s3_store, _ = get_s3_path(seg_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    input_key = f"s{scale}"
    with zarr.open(s3_store, mode="r") as f:
        shape = f[input_key].shape

    # Scale the coordinates according to the scale level.
    resolution = 0.38
    coordinates = syn_table[["z", "y", "x"]].values
    coordinates /= resolution
    coordinates /= (2 ** scale)
    coordinates = np.round(coordinates, 0).astype("int")

    ihc_ids = syn_table["matched_ihc"].values

    # Create the output.
    output = np.zeros(shape, dtype="uint16")
    mask = ball(radius).astype(bool)

    for coord, matched_ihc in tqdm(
        zip(coordinates, ihc_ids), total=len(coordinates), desc="Writing synapses to volume"
    ):
        bb = tuple(slice(c - radius, c + radius + 1) for c in coord)
        try:
            output[bb][mask] = matched_ihc
        except IndexError:
            print("Index error for", coord)
            continue

    # Write the output.
    out_folder = os.path.join(output_folder, cochlea, f"scale{scale}")
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{synapse_name}.tif")
    print("Writing synapses to", out_path)
    tifffile.imwrite(out_path, output, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--synapse_name", default="synapse_v3_ihc_v4")
    parser.add_argument("--reference_ihcs", default="IHC_v4")
    parser.add_argument("--max_dist", type=float, default=3.0)
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()

    export_synapse_detections(
        args.cochlea, args.scale, args.output_folder,
        args.synapse_name, args.reference_ihcs,
        args.max_dist, args.radius
    )


if __name__ == "__main__":
    main()
