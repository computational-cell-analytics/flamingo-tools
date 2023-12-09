import os
from glob import glob

import imageio.v3 as imageio
import pybdv

from typing import Optional, List, Dict


def _read_resolution_and_unit_flamingo(mdata_path):
    resolution = None
    with open(mdata_path, "r") as f:
        for line in f.readlines():
            line = line.strip().rstrip("\n")
            if line.startswith("Plane spacing"):
                resolution = float(line.split(" ")[-1])
                break
    if resolution is None:
        raise RuntimeError
    unit = "um"

    # FIXME dirty hack
    resolution = 4 * resolution

    # FIXME how do we get in-plane resolution, I don't believe it's always isotropic
    resolution = [resolution] * 3
    return resolution, unit


def get_global_metadata(metadata_paths, resolution, unit, metadata_type):
    if metadata_paths is not None:
        mdata_path = metadata_paths[0]
        resolution, unit = _read_resolution_and_unit_flamingo(mdata_path)

    if resolution is None:
        resolution = [1.0, 1.0, 1.0]
    if unit is None:
        unit = "pixel"

    return resolution, unit


# TODO expose scale factors as arguments
def convert_lightsheet_to_bdv(
    root: str,
    channel_folders: Dict[str, str],
    image_file_name_pattern: str,
    out_path: str,
    metadata_file_name_pattern: Optional[str] = None,
    metadata_root: Optional[str] = None,
    metadata_type: str = "flamingo",
    resolution: Optional[List[float]] = None,
    unit: Optional[str] = None,
) -> None:
    """This function converts the channels of one region/tile into a bdv-n5 file
    that can be read by BigDataViewer or BigStticher.

    Args:
        root: The folder that contains the channel folders.
        channel_folders: The list of channel folder names.
        file_name_pattern: The pattern for file names for the tifs that contain the per-channel data.
            This expects a placeholder 0%i for the index that refers to the channel.
            See the example 'convert_first_sample' below for details.
        out_path: Where to save the converted data.
        resolution: The resolution / physical size of one pixel.
        unit: The unit of the given resolution.
    """
    if metadata_type != "flamingo":
        raise ValueError

    # downsampling in ZYX
    # Isotropic donwsampling two times: [[2, 2, 2], [2, 2, 2]]
    # Anisotropic donwsampling two times: [[1, 2, 2], [1, 2, 2]]
    # First anisotropic donwsampling than isotropic [[1, 2, 2], [2, 2, 2]]
    scale_factors = [[2, 2, 2]] * 3
    n_threads = 8

    # Iterate over the channels
    for channel_id, (channel_name, channel_folder) in enumerate(channel_folders.items()):

        # Get all the image file paths for this channel.
        tile_pattern = os.path.join(root, channel_folder, image_file_name_pattern)
        file_paths = sorted(glob(tile_pattern))
        assert len(file_paths) > 0, tile_pattern

        # Get the corresponding
        if metadata_file_name_pattern is None:
            metadata_paths = None
        else:
            metadata_pattern = os.path.join(
                root if metadata_root is None else metadata_root,
                channel_folder, metadata_file_name_pattern
            )
            metadata_paths = sorted(glob(metadata_pattern))
            assert len(metadata_paths) == len(file_paths)

        resolution, unit = get_global_metadata(metadata_paths, resolution, unit, metadata_type)

        for tile_id, file_path in enumerate(file_paths):

            # TODO support memmap
            print("Loading data from tif ...")
            data = imageio.imread(file_path)
            print("done!")
            print("The data has the following shape:", data.shape)

            # TODO
            # transformation = read_tile_transformation()

            pybdv.make_bdv(
                data, out_path,
                downscale_factors=scale_factors, downscale_mode="mean",
                n_threads=n_threads,
                resolution=resolution, unit=unit,
                attributes={
                    "channel": {"id": channel_id, "name": channel_name}, "tile": {"id": tile_id},
                    "angle": {"id": 0}, "illumination": {"id": 0}
                },
                # transformation="",  # read the tile offset from metadata and apply it
            )
