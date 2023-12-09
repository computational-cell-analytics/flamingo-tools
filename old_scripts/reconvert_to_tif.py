import os

import imageio.v3 as imageio
import z5py


# select a lower scale and copy all relevant data
def reconvert_to_tif():
    input_file = "./converted/x.n5"
    out_root = "./converted/lightsheet_data_downscaled"
    fname_pattern = "S000_t000000_V000_R000%i_X000_Y000_C0%i_I0_D0_P04239.tif"
    n_setups = 12

    channel_folders = ["20230804_075941_MLR_136_1_L_561nm_PV", "20230804_093312_MLR_136_1_L_488nm_eYFP"]

    with z5py.File(input_file, "r") as f:
        for setup_id in range(1, n_setups + 1):

            if setup_id <= 6:
                channel = 0
                tile = setup_id - 1
            else:
                channel = 1
                tile = setup_id - 7

            # print(setup_id, ":", channel, tile)

            ds = f[f"setup{setup_id}"]
            data = ds[:]

            out_folder = os.path.join(out_root, channel_folders[channel])
            os.makedirs(out_folder, exist_ok=True)
            out_path = os.path.join(out_folder, fname_pattern % (tile, channel))

            imageio.imwrite(out_path, data)


def main():
    reconvert_to_tif()


if __name__ == "__main__":
    main()
