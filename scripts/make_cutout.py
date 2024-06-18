import os
import argparse
import json

from elf.io import open_file


def make_cutout(input_path, output_folder, coordinate):
    halo = 50, 512, 512

    with open_file(input_path, "r") as f:
        ds = f["raw"]
        shape = ds.shape
        bb = tuple(slice(max(ce - ha, 0), min(ce + ha, sh)) for ce, ha, sh in zip(coordinate, halo, shape))
        raw = ds[bb]

    pred_path = os.path.join(output_folder, "predictions.zarr")
    with open_file(pred_path, "r") as f:
        pred = f["prediction"][(slice(None),) + bb]

    seg_path = os.path.join(output_folder, "segmentation.zarr")
    with open_file(seg_path, "r") as f:
        seg = f["segmentation"][bb]

    with open_file("extracted.h5", "a") as f:
        f.create_dataset(raw, data=raw, compression="gzip")
        f.create_dataset(pred, data=pred, compression="gzip")
        f.create_dataset(seg, data=seg, compression="gzip")


def main():
    # 664,685,985
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_arguemnt("-c", "--coordinate", required=True)
    args = parser.parse_args()

    make_cutout(args.input_path, args.output_folder, json.loads(args.coordinate))


if __name__ == "__main__":
    main()
