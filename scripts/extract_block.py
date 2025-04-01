import os
import argparse
import numpy as np
import h5py
import z5py

"""
This script extracts data around an input center coordinate in a given ROI halo.
"""


def main(input_file, output_dir, input_key, resolution, coords, roi_halo):
    """

    :param str input_file: File path to input folder in n5 format
    :param str output_dir: output directory for saving cropped n5 file as <basename>_crop.n5
    :param str input_key: Key for accessing volume in n5 format, e.g. 'setup0/s0'
    :param float resolution: Resolution of input data in micrometer
    :param str coords: Center coordinates of extracted 3D volume in format 'z,y,x'
    :param str roi_halo: ROI halo of extracted 3D volume in format 'z,y,x'
    """

    coords =  [int(r) for r in coords.split(",")]
    roi_halo = [int(r) for r in roi_halo.split(",")]

    input_content = list(filter(None, input_file.split("/")))
    basename = "".join(input_content[-1].split(".")[:-1])
    input_dir = input_file.split(basename)[0]
    input_dir = os.path.abspath(input_dir)

    if "" == output_dir:
        output_dir = input_dir

    input_key = "setup0/timepoint0/s0"

    output_file = os.path.join(output_dir, basename + "_crop" + ".n5")

    #M_LR_000167_R, coords = '806,1042,1334', coords = (z, y, x) compared to MoBIE view

    coords = np.array(coords)
    coords = coords / resolution
    coords = np.round(coords).astype(np.int32)

    roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coords, roi_halo))

    with z5py.File(input_file, "r") as f:
        raw = f[input_key][roi]

    with z5py.File(output_file, "w") as f_out:
        f_out.create_dataset("raw", data=raw, compression="gzip")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('input', type=str, help="Input file in n5 format.")
    parser.add_argument('-o', "--output", type=str, default="", help="Output directory")
    parser.add_argument('-c', "--coord", type=str, required=True, help="3D coordinate in format 'z,y,x' as center of extracted block. Dimensions are inversed to view in MoBIE (x y z) -> (z y x)")

    parser.add_argument('-k', "--input_key", type=str, default="setup0/timepoint0/s0", help="Input key for data in input file")
    parser.add_argument('-r', "--resolution", type=float, default=0.38, help="Resolution of input in micrometer")

    parser.add_argument("--roi_halo", type=str, default="128,128,64", help="ROI halo around center coordinate")

    args = parser.parse_args()

    main(args.input, args.output, args.input_key, args.resolution, args.coord, args.roi_halo)
