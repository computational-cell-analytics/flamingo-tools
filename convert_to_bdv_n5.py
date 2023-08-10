import os
import pybdv
# import tifffile
import imageio.v3 as imageio


def convert_region_to_bdv_n5(root, channel_folders, file_name, out_path):
    scale_factors = [[2, 2, 2]] * 5
    n_threads = 8

    # TODO set correct resolution and scaling

    for setup_id, channel_folder in enumerate(channel_folders):
        file_path = os.path.join(root, channel_folder, file_name % setup_id)
        assert os.path.exists(file_path), file_path

        # memmapping doesn't work
        # print("Memory mapping ....")
        # data = tifffile.memmap(file_path, mode="r")
        # print("Memory mapped!!!")

        print("Loading data ...")
        data = imageio.imread(file_path)
        print("done!")
        print(data.shape)

        pybdv.make_bdv(
            data, out_path,
            downscale_factors=scale_factors, downscale_mode="mean",
            setup_id=setup_id, n_threads=n_threads
        )


def main():
    root = "/scratch1/users/pape41/data/moser/lightsheet/first_samples_lennart"
    channel_folders = [
        "20230804_040326_MLR_136_2_R_637nm_Myo7a",
        "20230804_031432_MLR_136_2_R_561nm_PV",
        "20230804_060847_MLR_136_2_R_488nm_eYFP",
    ]
    file_name_pattern = "S000_t000000_V000_R0000_X000_Y000_C0%i_I0_D0_P02995.tif"

    out_folder = os.path.join(root, "converted")
    os.makedirs(out_folder, exist_ok=True)
    name = "S000_t000000_V000_R0000_X000_Y000_C00_I0_D0_P02995.tif"
    out_path = os.path.join(out_folder, f"{name}.n5")

    convert_region_to_bdv_n5(root, channel_folders, file_name_pattern, out_path)


if __name__ == "__main__":
    main()
