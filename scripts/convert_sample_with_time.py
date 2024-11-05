from flamingo_tools.data_conversion import convert_lightsheet_to_bdv


def main():
    root = "/mnt/lustre-emmy-hdd/usr/u12086/data/flamingo/Platynereis-H2B-TL"
    output = "./output"
    convert_lightsheet_to_bdv(
        root, out_path=output,
        metadata_file_name_pattern="*_Settings.txt"
    )


main()
