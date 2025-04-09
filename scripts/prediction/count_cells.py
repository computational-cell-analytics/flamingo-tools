import argparse
import os
import sys

from elf.parallel import unique
from elf.io import open_file

sys.path.append("../..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Output directory containing segmentation.zarr")
    parser.add_argument("-m", "--min_size", type=int, default=1000, help="Minimal number of voxel size for counting object")
    args = parser.parse_args()

    seg_path = os.path.join(args.output_folder, "segmentation.zarr")
    seg_key = "segmentation"

    file = open_file(seg_path, mode='r')
    dataset = file[seg_key]

    ids, counts = unique(dataset, return_counts=True)

    # You can change the minimal size for objects to be counted here:
    min_size = args.min_size

    counts = counts[counts > min_size]
    print("Number of objects:", len(counts))

if __name__ == "__main__":
    main()
