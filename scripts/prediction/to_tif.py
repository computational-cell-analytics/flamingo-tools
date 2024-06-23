import argparse

import h5py
import tifffile
from elf.io import open_file

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-c", "--compression", action="store_true")

args = parser.parse_args()

with open_file(args.input, "r") as f:
    seg = f["segmentation"]
    seg.n_threads = 8
    print("Read segmentation ...")
    seg = seg[:].astype("uint32")


if seg.max() < 65000:
    seg = seg.astype("uint16")


print("Write segmentation ...")
if args.output.endswith(".h5"):
    with h5py.File(args.output, "a") as f:
        f.create_dataset("segmentation", data=seg, compression="gzip")
elif args.compression:
    tifffile.imwrite(args.output, seg, compression="zlib")
else:
    # Don't use compression, so that we can open this tif in Fiji.
    # Write as bigtiff.
    tifffile.imwrite(args.output, seg, bigtiff=True)
