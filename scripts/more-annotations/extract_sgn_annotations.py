import os
from flamingo_tools.extract_block_util import extract_block

RESOLUTION_LA_VISION = (1.887779, 1.887779, 3.000000)
RESOLUTION_FLAMINGO = (0.38, 0.38, 0.38)

POSITIONS = [
    [2451.991261845771, 2497.0312568466725, 504.00000000000017],
    [2364.0285060661868, 2104.541310616445, 684.2966391951725],
    [2579.872281689804, 2266.294057961108, 532.8474622712272],
    [2251.404321115024,1972.6189003459485,313.69577047550024],
]

EMPTY_POSITIONS = [
    [3091.354274253545, 1396.2702622881343, 443.21051449917223],
    [1052.8399693103918, 2180.579279395121, 437.81154147679354],
    [3621.8731222257875, 1602.0695390382377, 620.0517925327181],
]


def download_lavision_crops():
    input_path = "LaVision-M04/images/ome-zarr/PV.ome.zarr"
    input_key = "s0"
    output_key = None

    output_folder = "./LA_VISION_M04"
    os.makedirs(output_folder, exist_ok=True)
    for pos in POSITIONS:
        halo = [128, 128, 32]
        extract_block(
            input_path, pos, output_folder, input_key, output_key, RESOLUTION_LA_VISION, halo,
            tif=True, s3=True,
        )

    output_folder = "./LA_VISION_M04_empty"
    os.makedirs(output_folder, exist_ok=True)
    for pos in EMPTY_POSITIONS:
        halo = [128, 128, 32]
        extract_block(
            input_path, pos, output_folder, input_key, output_key, RESOLUTION_LA_VISION, halo,
            tif=True, s3=True,
        )


def downscale_segmentation():
    # Scale levels:
    # 0: 0.38
    # 1: 0.76
    # 2: 1.52
    cochleae = []


def main():
    download_lavision_crops()
    # downscale_segmentation()


if __name__ == "__main__":
    main()
