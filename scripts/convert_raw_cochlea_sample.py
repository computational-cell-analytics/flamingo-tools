from flamingo_tools.data_conversion import convert_lightsheet_to_bdv


def main():
    root = "/mnt/lustre-grete/usr/u12086/moser/lightsheet/M_LR_000172_R"
    output = "/mnt/lustre-grete/usr/u12086/moser/lightsheet/M_LR_000172_R/converted.n5"
    convert_lightsheet_to_bdv(
        root, out_path=output, file_ext=".raw",
        metadata_file_name_pattern="*_Settings.txt"
    )


main()
