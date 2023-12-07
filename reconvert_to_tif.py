import os
from glob import glob
from pathlib import Path

import imageio.v2 as imageio
import z5py


# select a lower scale and copy all relevant data
def reconvert_to_tif(input_path, output_root, scale):
    fname = Path(input_path).stem
    with z5py.File(input_path, "r") as f:
        n_setups = len(f)
        for setup_id in range(n_setups):
            channel = f"C0{setup_id}"
            output_folder = os.path.join(output_root, channel)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"{fname}_{channel}.tif")

            data = f[f"setup{setup_id}/timepoint0/s{scale}"][:]
            print("Copy data of shape", data.shape, "to", output_path)
            imageio.volwrite(output_path, data)


def main():
    input_paths = glob(
        "/scratch-grete/usr/nimcpape/data/moser/lightsheet/data-aleyna/*.n5"
    )
    output_root = "/scratch-grete/usr/nimcpape/data/moser/lightsheet/data-aleyna/tif"
    scale = 3

    for input_path in input_paths:
        reconvert_to_tif(input_path, output_root, scale)


if __name__ == "__main__":
    main()
