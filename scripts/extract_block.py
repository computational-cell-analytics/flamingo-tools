import os
import argparse
import numpy as np
import zarr

import flamingo_tools.s3_utils as s3_utils

"""
This script extracts data around an input center coordinate in a given ROI halo.

The support for using an S3 bucket is currently limited to the lightsheet-cochlea bucket with the endpoint url https://s3.fs.gwdg.de.
If more use cases appear, the script will be generalized.
The usage requires the export of the access and the secret access key within the environment before executing the script.
run the following commands in the shell of your choice, or add them to your ~/.bashrc:
export AWS_ACCESS_KEY_ID=<access key>
export AWS_SECRET_ACCESS_KEY=<secret access key>
"""


def main(
    input_file, output_dir, coords, input_key, resolution, roi_halo,
    s3, s3_credentials, s3_bucket_name, s3_service_endpoint,
    ):
    """

    :param str input_file: File path to input folder in n5 format
    :param str output_dir: output directory for saving cropped n5 file as <basename>_crop.n5
    :param str input_key: Key for accessing volume in n5 format, e.g. 'setup0/s0'
    :param float resolution: Resolution of input data in micrometer
    :param str coords: Center coordinates of extracted 3D volume in format 'x,y,z'
    :param str roi_halo: ROI halo of extracted 3D volume in format 'x,y,z'
    :param bool s3: Flag for using an S3 bucket
    :param str s3_credentials: Path to file containing S3 credentials
    :param str s3_bucket_name: S3 bucket name. Optional if BUCKET_NAME has been exported
    :param str s3_service_endpoint: S3 service endpoint. Optional if SERVICE_ENDPOINT has been exported
    """

    coords =  [int(r) for r in coords.split(",")]
    roi_halo = [int(r) for r in roi_halo.split(",")]

    coord_string = "-".join([str(c) for c in coords])

    # Dimensions are inversed to view in MoBIE (x y z) -> (z y x)
    coords.reverse()
    roi_halo.reverse()

    input_content = list(filter(None, input_file.split("/")))

    if s3:
        basename = input_content[0] + "_" + input_content[-1].split(".")[0]
    else:
        basename = "".join(input_content[-1].split(".")[:-1])

    input_dir = input_file.split(basename)[0]
    input_dir = os.path.abspath(input_dir)

    if output_dir == "":
        output_dir = input_dir

    output_file = os.path.join(output_dir, basename + "_crop_" + coord_string + ".n5")

    coords = np.array(coords)
    coords = coords / resolution
    coords = np.round(coords).astype(np.int32)

    roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coords, roi_halo))

    if s3:
        s3_path, fs = s3_utils.get_s3_path(input_file, bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint, credential_file=s3_credentials)

        with zarr.open(s3_path, mode="r") as f:
            raw = f[input_key][roi]

    else:
        with zarr.open(input_file, mode="r") as f:
            raw = f[input_key][roi]

    with zarr.open(output_file, mode="w") as f_out:
        f_out.create_dataset("raw", data=raw, compression="gzip")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('input', type=str, help="Input file in n5 format.")
    parser.add_argument('-o', "--output", type=str, default="", help="Output directory")
    parser.add_argument('-c', "--coord", type=str, required=True, help="3D coordinate in format 'x,y,z' as center of extracted block.")

    parser.add_argument('-k', "--input_key", type=str, default="setup0/timepoint0/s0", help="Input key for data in input file")
    parser.add_argument('-r', "--resolution", type=float, default=0.38, help="Resolution of input in micrometer")

    parser.add_argument("--roi_halo", type=str, default="128,128,64", help="ROI halo around center coordinate in format 'x,y,z'")

    parser.add_argument("--s3", action="store_true", help="Use S3 bucket")
    parser.add_argument("--s3_credentials", default=None, help="Input file containing S3 credentials")
    parser.add_argument("--s3_bucket_name", default=None, help="S3 bucket name")
    parser.add_argument("--s3_service_endpoint", default=None, help="S3 service endpoint")

    args = parser.parse_args()

    main(
        args.input, args.output, args.coord, args.input_key, args.resolution, args.roi_halo,
        args.s3, args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )
