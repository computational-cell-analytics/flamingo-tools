import os

import imageio.v3 as imageio
from skimage.data import binary_blobs


# TODO add metadata
def create_test_data(root: str, size: int = 256, n_channels: int = 2, n_tiles: int = 4) -> None:
    """Create test data in the flamingo data format.

    Args:
        root: Directory for saving the data.
        size: The axis length for the data.
        n_channels The number of channels to create:
        n_tiles: The number of tiles to create.
    """
    channel_folders = [f"channel{chan_id}" for chan_id in range(n_channels)]
    file_name_pattern = "volume_R%i_C%i_I0.tif"
    for chan_id, channel_folder in enumerate(channel_folders):
        out_folder = os.path.join(root, channel_folder)
        os.makedirs(out_folder, exist_ok=True)
        for tile_id in range(n_tiles):
            out_path = os.path.join(out_folder, file_name_pattern % (tile_id, chan_id))
            data = binary_blobs(size, n_dim=3).astype("uint8") * 255
            imageio.imwrite(out_path, data)
