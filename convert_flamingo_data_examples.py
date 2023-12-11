from flamingo_tools import convert_lightsheet_to_bdv


def convert_synthetic_data():
    """Script with settings to convert the synthetic data created via
    `create_synthetic_data.py`.
    """

    # Root folder with the data.
    root = "./synthetic_data"

    # Names and folder names for the channels to be converted.
    channel_folders = {
        "c1": "channel0",
        "c2": "channel1"
    }

    # Name patterns for the image data and text files with metadata.
    image_file_name_pattern = "*.tif"

    # Filepath where the converted data should be saved.
    out_path = "./synthetic_data/converted.n5"

    # Run the conversion.
    convert_lightsheet_to_bdv(
        root, channel_folders, image_file_name_pattern, out_path,
    )


def convert_sample_data_moser():
    """Script with settings to convert sample flamingo data
    shared by the Moser group.
    """

    # Root folder with the data.
    root = "./converted/downsampled_tifs"

    # Names and folder names for the channels to be converted.
    channel_folders = {
        "PV": "20230804_075941_MLR_136_1_L_561nm_PV",
        "eYFP": "20230804_093312_MLR_136_1_L_488nm_eYFP"
    }

    # Name patterns for the image data and text files with metadata.
    image_file_name_pattern = "S000_t000000_V000_R000*.tif"
    metadata_file_name_pattern = "S000_t000000_V000_R000*_Settings.txt"

    # Filepath where the converted data should be saved.
    out_path = "./converted/converted.n5"

    # Run the conversion.
    convert_lightsheet_to_bdv(
        root, channel_folders, image_file_name_pattern, out_path,
        metadata_file_name_pattern=metadata_file_name_pattern
    )


def main():
    convert_synthetic_data()
    # convert_sample_data_moser()


if __name__ == "__main__":
    main()
