import os
from glob import glob
from subprocess import run

import zarr
from flamingo_tools.data_conversion import convert_lightsheet_to_bdv

ROOT = "/mnt/lustre-emmy-hdd/usr/u12086/data/flamingo"


def convert_to_ome_zarr_v2(name):
    input_root = os.path.join(ROOT, name)
    assert os.path.exists(input_root)

    output_root = os.path.join(ROOT, "ngff-v2")
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{name}.ome.zarr")

    convert_lightsheet_to_bdv(input_root, out_path=output_path)


def convert_to_ome_zarr_v3(name):
    input_path = os.path.join(ROOT, "ngff-v2", f"{name}.ome.zarr")

    output_root = os.path.join(ROOT, "ngff-v3")
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{name}.ome.zarr")

    f_in = zarr.v2.open(store=input_path, mode="r")
    f_out = zarr.open_group(store=output_path, mode="a")

    setup_folders = sorted(glob(os.path.join(input_path, "setup*")))
    for sfolder in setup_folders:
        setup = os.path.basename(sfolder)
        f_out.create_group(name=setup)

        attrs = {k: v for k, v in f_in[setup].attrs.items()}
        f_out[setup].attrs.update(attrs)

        # Copy over the attributes for this set-up.
        timepoint_folders = sorted(glob(os.path.join(sfolder, "timepoint*")))
        for tfolder in timepoint_folders:
            timepoint = os.path.basename(tfolder)
            print("Converting", setup, timepoint)
            out = os.path.join(output_path, setup, timepoint)
            cmd = [
                "ome2024-ngff-challenge", "resave", "--cc-by", tfolder, out,
                "--output-overwrite", "--output-shards=512,512,512"
            ]
            run(cmd)


def main():
    # name = "Platynereis-H2B-TL"
    # name = "Zebrafish-XSPIM-multiview"
    name = "Zebrafish-H2B-short-timelapse"
    convert_to_ome_zarr_v2(name)
    convert_to_ome_zarr_v3(name)


if __name__ == "__main__":
    main()
