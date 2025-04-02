import os, sys
import argparse
import pybdv
import imageio.v3 as imageio


def main(input_path, output_path):
    """
    Convert tif file to n5 format.
    If no output_path is supplied, the output file is created in the same directory as the input.

    :param str input_path: Input tif
    :param str output_path: Output path for n5 format
    """
    if not os.path.isfile(input_path):
        sys.exit("Input file does not exist.")

    if input_path.split(".")[-1] not in ["TIFF", "TIF", "tiff", "tif"]:
        sys.exit("Input file must be in tif format.")

    basename = "".join(input_path.split("/")[-1].split(".")[:-1])
    input_dir = input_path.split(basename)[0]
    input_dir = os.path.abspath(input_dir)

    if "" == output_path:
        output_path = os.path.join(input_dir, basename + ".n5")
    img = imageio.imread(input_path)
    pybdv.make_bdv(img, output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to transform file from tif into n5 format.")

    parser.add_argument('input', type=str, help="Input file")
    parser.add_argument('-o', "--output", type=str, default="", help="Output file. Default: <basename>.n5")

    args = parser.parse_args()

    main(args.input, args.output)