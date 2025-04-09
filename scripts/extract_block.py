import os
import argparse
import numpy as np
import z5py
import zarr

import s3fs

"""
This script extracts data around an input center coordinate in a given ROI halo.

The support for using an S3 bucket is currently limited to the lightsheet-cochlea bucket with the endpoint url https://s3.fs.gwdg.de.
If more use cases appear, the script will be generalized.
The usage requires the export of the access and the secret access key within the environment before executing the script.
run the following commands in the shell of your choice, or add them to your ~/.bashrc:
export AWS_ACCESS_KEY_ID=<access key>
export AWS_SECRET_ACCESS_KEY=<secret access key>
"""


def main(input_file, output_dir, input_key, resolution, coords, roi_halo, s3):
    """

    :param str input_file: File path to input folder in n5 format
    :param str output_dir: output directory for saving cropped n5 file as <basename>_crop.n5
    :param str input_key: Key for accessing volume in n5 format, e.g. 'setup0/s0'
    :param float resolution: Resolution of input data in micrometer
    :param str coords: Center coordinates of extracted 3D volume in format 'x,y,z'
    :param str roi_halo: ROI halo of extracted 3D volume in format 'x,y,z'
    :param bool s3: Flag for using an S3 bucket
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

        # Define S3 bucket and OME-Zarr dataset path

        bucket_name = "cochlea-lightsheet"
        zarr_path = f"{bucket_name}/{input_file}"

        # Create an S3 filesystem
        fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": "https://s3.fs.gwdg.de"},
            anon=False
        )

        if not fs.exists(zarr_path):
            print("Error: Path does not exist!")

        # Open the OME-Zarr dataset
        store = zarr.storage.FSStore(zarr_path, fs=fs)
        print(f"Opening file {zarr_path} from the S3 bucket.")

        with zarr.open(store, mode="r") as f:
            raw = f[input_key][roi]

    else:
        with z5py.File(input_file, "r") as f:
            raw = f[input_key][roi]

    with z5py.File(output_file, "w") as f_out:
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

    args = parser.parse_args()

    main(args.input, args.output, args.input_key, args.resolution, args.coord, args.roi_halo, args.s3)
