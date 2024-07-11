import argparse
import os
import sys

import z5py

sys.path.append("../..")


def main():
    from flamingo_tools.segmentation import filter_isolated_objects

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder", required=True)
    args = parser.parse_args()

    seg_path = os.path.join(args.output_folder, "segmentation.zarr")
    seg_key = "segmentation"

    with z5py.File(seg_path, "r") as f:
        segmentation = f[seg_key][:]

    seg_filtered, n_pre, n_post = filter_isolated_objects(segmentation)

    with z5py.File(seg_path, "a") as f:
        chunks = f[seg_key].chunks
        f.create_dataset(
            "segmentation_postprocessed", data=seg_filtered, compression="gzip",
            chunks=chunks, dtype=seg_filtered.dtype
        )


if __name__ == "__main__":
    main()
