# Create additional top-level metadata.

import argparse
import os
import json
from glob import glob


def get_series(path):
    setups = sorted(glob(os.path.join(path, "**/timepoint*")))
    setups = [os.path.relpath(p, path) for p in setups]
    return setups


def create_metadata(path):
    # TOP LEVEL METADATA
    bf_to_raw = {
        "attributes": {
            "ome": {
                "version": "0.5",
                "bioformats2raw.layout": 3
            }
        },
        "zarr_format": 3,
        "node_type": "group",
    }
    meta_path = os.path.join(path, "zarr.json")

    # This can be safely over-written.
    with open(meta_path, "w") as f:
        json.dump(bf_to_raw, f)

    # OME METADATA
    series = get_series(path)
    ome_metadata = {
        "attributes": {
            "ome": {
                "version": "0.5",
                "series": series
            }
        },
        "zarr_format": 3,
        "node_type": "group",
    }
    meta_folder = os.path.join(path, "OME")
    os.makedirs(meta_folder, exist_ok=True)
    meta_path = os.path.join(meta_folder, "zarr.json")
    with open(meta_path, "w") as f:
        json.dump(ome_metadata, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    create_metadata(args.path)


if __name__ == "__main__":
    main()
