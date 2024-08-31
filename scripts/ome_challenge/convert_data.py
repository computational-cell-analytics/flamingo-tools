import os
import sys

ROOT = "/mnt/lustre-emmy-hdd/usr/u12086/data/flamingo"


def convert_to_ome_zarr_v2(name):
    sys.path.append("../..")
    from flamingo_tools.data_conversion import convert_lightsheet_to_bdv

    input_root = os.path.join(ROOT, name)
    assert os.path.exists(input_root)

    output_root = os.path.join(ROOT, "ngff-v2")
    os.makedirs(output_root, exist_ok=True)

    output_path = os.path.join(output_root, f"{name}.ome.zarr")

    # Number of timepoints:
    # ntp = 10
    ntp = 1  # for testing

    channel_folders = {f"t{tp:02}": "" for tp in range(ntp)}
    convert_lightsheet_to_bdv(
        input_root, channel_folders, image_file_name_pattern="*_t000000_*_C01_I0_*.tif",
        out_path=output_path,
    )


def convert_to_ome_zarr_v3(name):
    pass


def main():
    name = "Platynereis-H2B-TL"
    convert_to_ome_zarr_v2(name)


if __name__ == "__main__":
    main()
