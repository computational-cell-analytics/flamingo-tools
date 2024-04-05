import argparse

import imageio.v3 as imageio
from elf.io import open_file

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)

args = parser.parse_args()

with open_file(args.input, "r") as f:
    seg = f["segmentation"]
    seg.n_threads = 8
    seg = seg[:]

imageio.imwrite(args.output, seg, compression="zlib")
