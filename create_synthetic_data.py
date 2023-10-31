import os
import imageio.v3 as imageio
from skimage.data import binary_blobs


def create_synthetic_data():
    root = "./synthetic_data"
    channel_folders = ["channel1", "channel2", "channel3"]
    file_name_pattern = "volume_C0%i.tif"

    size = 512
    for i, channel_folder in enumerate(channel_folders):
        out_folder = os.path.join(root, channel_folder)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, file_name_pattern % i)
        data = binary_blobs(size, n_dim=3).astype("uint8") * 255
        imageio.imwrite(out_path, data)


def main():
    create_synthetic_data()


if __name__ == "__main__":
    main()
