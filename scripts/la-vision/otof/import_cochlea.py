import os
import h5py
from mobie import add_image
from mobie.metadata import read_dataset_metadata

INPUT_PATH = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LA_VISION_OTOF/Test_FreeRotate_0-40-59_PRO82_OtofKO-23R_p24_chCR-488_rbOtof-647_UltraII_C00_xyz.ims"  # noqa
MOBIE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
DS_NAME = "LaVision-OTOF"
RESOLUTION = (3.0, 1.887779, 1.887779)


# Channels: "chCR-488_rbOtof-647"
def add_otof():
    channel_names = ("CR", "rbOtof")
    channel_keys = [
        "/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
        "/DataSet/ResolutionLevel 0/TimePoint 0/Channel 1/Data"
    ]

    scale_factors = 4 * [[2, 2, 2]]
    chunks = (96, 96, 96)

    for channel_key, channel_name in zip(channel_keys, channel_names):
        mobie_ds_folder = os.path.join(MOBIE_ROOT, DS_NAME)
        ds_metadata = read_dataset_metadata(mobie_ds_folder)
        if channel_name in ds_metadata.get("sources", {}):
            print(channel_name, "is already in MoBIE")
            continue

        print("Load image data ...")
        with h5py.File(INPUT_PATH, "r") as f:
            input_data = f[channel_key][:]
        print(input_data.shape)
        add_image(
            input_path=input_data, input_key=None, root=MOBIE_ROOT,
            dataset_name=DS_NAME, image_name=channel_name, resolution=RESOLUTION,
            scale_factors=scale_factors, chunks=chunks, unit="micrometer", use_memmap=False,
        )


def main():
    add_otof()


if __name__ == "__main__":
    main()
