import os
from typing import Tuple

import imageio.v3 as imageio
import requests
from skimage.data import binary_blobs, cells3d
from skimage.measure import label

from .segmentation.postprocessing import compute_table_on_the_fly

SEGMENTATION_URL = "https://owncloud.gwdg.de/index.php/s/kwoGRYiJRRrswgw/download"


def get_test_volume_and_segmentation(folder: str) -> Tuple[str, str, str]:
    """Download a small volume with nuclei and corresponding segmentation.

    Args:
        folder: The test data folder. The data will be downloaded to this folder.

    Returns:
        The path to the image, stored as tif.
        The path to the segmentation, stored as tif.
        The path to the segmentation table, stored as tsv.
    """
    os.makedirs(folder, exist_ok=True)

    segmentation_path = os.path.join(folder, "segmentation.tif")
    resp = requests.get(SEGMENTATION_URL)
    resp.raise_for_status()

    with open(segmentation_path, "wb") as f:
        f.write(resp.content)

    nuclei = cells3d()[20:40, 1]
    segmentation = imageio.imread(segmentation_path)
    assert nuclei.shape == segmentation.shape

    image_path = os.path.join(folder, "image.tif")
    imageio.imwrite(image_path, nuclei)

    table_path = os.path.join(folder, "default.tsv")
    table = compute_table_on_the_fly(segmentation, resolution=0.38)
    table.to_csv(table_path, sep="\t", index=False)

    return image_path, segmentation_path, table_path


def create_image_data_and_segmentation(folder: str, size: int = 256) -> Tuple[str, str, str]:
    """Create test data containing an image, a corresponding segmentation and segmentation table.

    Args:
        folder: The test data folder. The data will be written to this folder.

    Returns:
        The path to the image, stored as tif.
        The path to the segmentation, stored as tif.
        The path to the segmentation table, stored as tsv.
    """
    os.makedirs(folder, exist_ok=True)
    data = binary_blobs(size, n_dim=3).astype("uint8") * 255
    seg = label(data)

    image_path = os.path.join(folder, "image.tif")
    segmentation_path = os.path.join(folder, "segmentation.tif")
    imageio.imwrite(image_path, data)
    imageio.imwrite(segmentation_path, seg)

    table_path = os.path.join(folder, "default.tsv")
    table = compute_table_on_the_fly(seg, resolution=0.38)
    table.to_csv(table_path, sep="\t", index=False)

    return image_path, segmentation_path, table_path


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


def sample_data_pv():
    pass


def sample_data_vglut3():
    pass


def sample_data_ctbp2():
    pass
