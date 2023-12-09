from flamingo_tools import convert_lightsheet_to_bdv


def convert_sample_data():
    root = "./converted/lightsheet_data_downscaled"
    channel_folders = {
        "PV": "20230804_075941_MLR_136_1_L_561nm_PV",
        "eYFP": "20230804_093312_MLR_136_1_L_488nm_eYFP"
    }
    image_file_name_pattern = "S000_t000000_V000_R000*.tif"
    metadata_file_name_pattern = "S000_t000000_V000_R000*_Settings.txt"

    out_path = "./converted/converted.n5"

    convert_lightsheet_to_bdv(
        root, channel_folders, image_file_name_pattern, out_path,
        metadata_file_name_pattern=metadata_file_name_pattern
    )


def main():
    convert_sample_data()


if __name__ == "__main__":
    main()
