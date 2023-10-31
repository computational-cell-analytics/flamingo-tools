import os
from typing import List

import imageio.v3 as imageio
import pybdv


def convert_region_to_bdv_n5(
    root: str,
    channel_folders: List[str],
    file_name_pattern: str,
    out_path: str,
    resolution: List[float] = [1.0, 1.0, 1.0],
    unit: str = "pixel",
):
    """This function converts the channels of one region/tile into a bdv-n5 file
    that can be read by BigDataViewer or BigStticher.

    Args:
        root: The folder that contains the channel folders.
        channel_folders: The list of channel folder names.
        file_name_pattern: The pattern for file names for the tifs that contain the per-channel data.
            This expects a placeholder 0%i for the index that refers to the channel.
            See the example 'convert_first_sample' below for details.
        out_path: Where to save the converted data.
        resolution: The resolution / physical size of one pixel.
        unit: The unit of the given resolutio. (Most likely 'micrometer').
    """
    scale_factors = [[2, 2, 2]] * 5
    n_threads = 8

    for setup_id, channel_folder in enumerate(channel_folders):
        file_path = os.path.join(root, channel_folder, file_name_pattern % setup_id)
        assert os.path.exists(file_path), file_path

        print("Loading data from tif ...")
        data = imageio.imread(file_path)
        print("done!")
        print("The data has the following shape:", data.shape)

        pybdv.make_bdv(
            data, out_path,
            downscale_factors=scale_factors, downscale_mode="mean",
            setup_id=setup_id, n_threads=n_threads,
            resolution=resolution, unit=unit,
        )


def convert_first_sample():
    """Example for using 'convert_region_to_bdv_n5' for converting actual data from Lennart.
    This was used for converting the data I have shared.
    """

    # Root is the folder where all data is stored
    # (here the folder I have copied the data too for my tests)
    root = "/scratch1/users/pape41/data/moser/lightsheet/first_samples_lennart"
    # And these are the names of the channels that contain the tif files for a channel and given region.
    channel_folders = [
        "20230804_040326_MLR_136_2_R_637nm_Myo7a",
        "20230804_031432_MLR_136_2_R_561nm_PV",
        "20230804_060847_MLR_136_2_R_488nm_eYFP",
    ]
    # And this is the name pattern for the tifs holding the per channel data.
    # '0%i' is a placeholder, for the channels that are called C00, C01 and C02 respectively.
    file_name_pattern = "S000_t000000_V000_R0000_X000_Y000_C0%i_I0_D0_P02995.tif"

    # Here we create the folder where the data will be stored.
    out_folder = os.path.join(root, "converted")
    os.makedirs(out_folder, exist_ok=True)

    # And then give the path to the output file.
    # In this case we use the same naming pattern as above, but remove the channel suffix,
    # since this file will contain all three channels.
    name = "S000_t000000_V000_R0000_X000_Y000_I0_D0_P02995"
    out_path = os.path.join(out_folder, f"{name}.n5")

    # TODO enter the correct values for the resolution and unit here.
    # I have kept dummy values for this since I don't know it ...
    resolution = [1.0, 1.0, 1.0]
    unit = "pixel"

    # Now we call the convertsion function with all necessary information.
    convert_region_to_bdv_n5(
        root, channel_folders, file_name_pattern, out_path,
        resolution=resolution, unit=unit,
    )


def convert_synthetic_data():
    """Example for using 'convert_region_to_bdv_n5' for converting synthetic data.
    The synthetic data can be created with 'create_synthetic_data.py'.
    """
    root = "./synthetic_data"
    channel_folders = ["channel1", "channel2", "channel3"]
    file_name_pattern = "volume_C0%i.tif"

    out_path = "./synthetic_data/synthetic_data.n5"
    convert_region_to_bdv_n5(root, channel_folders, file_name_pattern, out_path)


def main():
    # convert_first_sample()
    convert_synthetic_data()


if __name__ == "__main__":
    main()
