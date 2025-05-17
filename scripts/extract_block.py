"""This script extracts data around an input center coordinate in a given ROI halo.
"""
import argparse
import json
import os
from typing import Optional, List

import imageio.v3 as imageio
import numpy as np
import zarr

import flamingo_tools.s3_utils as s3_utils


def main(
    input_path: str,
    coords: List[int],
    output_dir: str = None,
    input_key: str = "setup0/timepoint0/s0",
    output_key: Optional[str] = None,
    resolution: float = 0.38,
    roi_halo: List[int] = [128, 128, 64],
    tif: bool = False,
    s3: Optional[bool] = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Extract block around coordinate from input data according to a given halo.
    Either from a local file or from an S3 bucket.

    Args:
        input_path: Input folder in n5 / ome-zarr format.
        coords: Center coordinates of extracted 3D volume.
        output_dir: Output directory for saving output as <basename>_crop.n5. Default: input directory.
        input_key: Input key for data in input file.
        output_key: Output key for data in n5 output or used as suffix for tif output.
        roi_halo: ROI halo of extracted 3D volume.
        tif: Flag for tif output
        s3: Flag for considering input_path for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
    """
    coords = [int(round(c)) for c in coords]
    coord_string = "-".join([str(c).zfill(4) for c in coords])

    # Dimensions are inversed to view in MoBIE (x y z) -> (z y x)
    coords.reverse()
    roi_halo.reverse()

    input_content = list(filter(None, input_path.split("/")))

    if s3:
        image_name = input_content[-1].split(".")[0]
        if len(image_name.split("_")) > 1:
            resized_suffix = "_resized"
            image_prefix = image_name.split("_")[0]
        else:
            resized_suffix = ""
            image_prefix = image_name
        basename = input_content[0] + resized_suffix
    else:
        basename = "".join(input_content[-1].split(".")[:-1])

    input_dir = input_path.split(basename)[0]
    input_dir = os.path.abspath(input_dir)

    if output_dir == "":
        output_dir = input_dir

    if tif:
        if output_key is None:
            output_name = basename + "_crop_" + coord_string + "_" + image_prefix + ".tif"
        else:
            output_name = basename + "_" + image_prefix + "_crop_" + coord_string + "_" + output_key + ".tif"

        output_file = os.path.join(output_dir, output_name)
    else:
        output_key = "raw" if output_key is None else output_key
        output_file = os.path.join(output_dir, basename + "_crop_" + coord_string + ".n5")

    coords = np.array(coords)
    coords = coords / resolution
    coords = np.round(coords).astype(np.int32)

    roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coords, roi_halo))

    if s3:
        s3_path, fs = s3_utils.get_s3_path(input_path, bucket_name=s3_bucket_name,
                                           service_endpoint=s3_service_endpoint, credential_file=s3_credentials)

        with zarr.open(s3_path, mode="r") as f:
            raw = f[input_key][roi]

    else:
        with zarr.open(input_path, mode="r") as f:
            raw = f[input_key][roi]

    if tif:
        imageio.imwrite(output_file, raw, compression="zlib")
    else:
        with zarr.open(output_file, mode="w") as f_out:
            f_out.create_dataset(output_key, data=raw, compression="gzip")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('-i', '--input', type=str, required=True, help="Input file in n5 / ome-zarr format.")
    parser.add_argument('-o', "--output", type=str, default="", help="Output directory.")
    parser.add_argument('-c', "--coord", type=str, required=True,
                        help="3D coordinate as center of extracted block, json-encoded.")

    parser.add_argument('-k', "--input_key", type=str, default="setup0/timepoint0/s0",
                        help="Input key for data in input file.")
    parser.add_argument("--output_key", type=str, default=None,
                        help="Output key for data in output file.")
    parser.add_argument('-r', "--resolution", type=float, default=0.38, help="Resolution of input in micrometer.")
    parser.add_argument("--roi_halo", type=str, default="[128,128,64]",
                        help="ROI halo around center coordinate, json-encoded.")
    parser.add_argument("--tif", action="store_true", help="Store output as tif file.")

    parser.add_argument("--s3", action="store_true", help="Use S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    coords = json.loads(args.coord)
    roi_halo = json.loads(args.roi_halo)

    main(
        args.input, coords, args.output, args.input_key, args.output_key, args.resolution, roi_halo,
        args.tif, args.s3, args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )
