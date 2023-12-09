import multiprocessing as mp
import os

from glob import glob
from typing import Optional, List, Dict

import numpy as np
import pybdv
import tifffile


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

    # FIXME how do we get in-plane resolution, I don't believe it's always isotropic
    resolution = [resolution] * 3
    return resolution, unit


def _read_start_position_flamingo(path):
    at_start = False
    start_x, start_y, start_z = None, None, None

    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip().rstrip("\n")
            if line.startswith("<Start Position>"):
                at_start = True

            if at_start and line.startswith("X"):
                start_x = float(line.split(" ")[-1])
            if at_start and line.startswith("Y"):
                start_y = float(line.split(" ")[-1])
            if at_start and line.startswith("Z"):
                start_z = float(line.split(" ")[-1])

            if (start_x is not None) and (start_y is not None) and (start_z is not None):
                break

    assert (start_x is not None) and (start_y is not None) and (start_z is not None)
    start_position = [start_x, start_y, start_z]
    return start_position


def read_metadata_flamingo(metadata_paths, center_tiles):
    start_positions = []
    resolution, unit = None, None
    for path in metadata_paths:
        resolution, unit = _read_resolution_and_unit_flamingo(path)
        start_position = _read_start_position_flamingo(path)
        start_positions.append(start_position)

    start_positions = np.array(start_positions)
    offset = np.min(start_positions, axis=0) if center_tiles else np.array([0.0, 0.0, 0.0])

    def _pos_to_trafo(pos):
        pos -= offset
        # FIXME: dirty hack
        scale = 4
        # The calibration: scale factors on the diagonals.
        calib_trafo = [
            scale * resolution[0], 0.0, 0.0, 0.0,
            0.0, scale * resolution[1], 0.0, 0.0,
            0.0, 0.0, scale * resolution[2], 0.0,
        ]
        # The translation to the grid position.
        # Note that the translations are given in mm,
        # so they need to multiplied by a factor of 1000
        # to match the resolution given in microns.
        grid_trafo = [
            1.0, 0.0, 0.0, scale * pos[0] * 1000,
            0.0, 1.0, 0.0, scale * pos[1] * 1000,
            0.0, 0.0, 1.0, scale * pos[2] * 1000,
        ]
        trafo = {
            "Translation to Regular Grid": grid_trafo,
            "Calibration": calib_trafo,
        }
        return trafo

    transformations = [
        _pos_to_trafo(pos) for pos in start_positions
    ]
    # We have to reverse the resolution because pybdv expects ZYX.
    return resolution[::-1], unit, transformations


# TODO derive the scale factors from the shape
def derive_scale_factors(shape):
    # downsampling in ZYX
    # Isotropic donwsampling two times: [[2, 2, 2], [2, 2, 2]]
    # Anisotropic donwsampling two times: [[1, 2, 2], [1, 2, 2]]
    # First anisotropic donwsampling than isotropic [[1, 2, 2], [2, 2, 2]]
    scale_factors = [[2, 2, 2]] * 3
    return scale_factors


def convert_lightsheet_to_bdv(
    root: str,
    channel_folders: Dict[str, str],
    image_file_name_pattern: str,
    out_path: str,
    metadata_file_name_pattern: Optional[str] = None,
    metadata_root: Optional[str] = None,
    metadata_type: str = "flamingo",
    center_tiles: bool = True,
    resolution: Optional[List[float]] = None,
    unit: Optional[str] = None,
    scale_factors: Optional[List[List[int]]] = None,
    n_threads: Optional[int] = None,
) -> None:
    """This function converts the channels of one region/tile into a bdv-n5 file
    that can be read by BigDataViewer or BigStticher.

    Args:
        root: The folder that contains the channel folders.
        channel_folders: The list of channel folder names.
        image_file_name_pattern: The pattern for file names for the tifs that contain the per-channel data.
            This expects a placeholder 0%i for the index that refers to the channel.
            See the example 'convert_first_sample' below for details.
        out_path: Where to save the converted data.
        metadata_file_name_pattern:
        metadata_type:
        center_tiles:
        resolution: The resolution / physical size of one pixel.
        unit: The unit of the given resolution.
        scale_factors:
        n_threads:
    """
    if metadata_type != "flamingo":
        raise ValueError(f"Invalid metadata type: {metadata_type}.")
    if n_threads is None:
        n_threads = mp.cpu_count()

    # Iterate over the channels
    for channel_id, (channel_name, channel_folder) in enumerate(channel_folders.items()):

        # Get all the image file paths for this channel.
        tile_pattern = os.path.join(root, channel_folder, image_file_name_pattern)
        file_paths = sorted(glob(tile_pattern))
        assert len(file_paths) > 0, tile_pattern

        # Read the metadata if it was given.
        if metadata_file_name_pattern is None:  # No metadata given.
            # We don't use any tile transformation.
            tile_transformations = [None] * len(file_paths)
            # Set resolution and unit to their default values if they were not passed.
            if resolution is None:
                resolution = [1.0, 1.0, 1.0]
            if unit is None:
                unit = "pixel"

        else:  # We have metadata and read it.
            metadata_pattern = os.path.join(
                root if metadata_root is None else metadata_root,
                channel_folder, metadata_file_name_pattern
            )
            metadata_paths = sorted(glob(metadata_pattern))
            assert len(metadata_paths) == len(file_paths)
            resolution, unit, tile_transformations = read_metadata_flamingo(metadata_paths, center_tiles)

        for tile_id, (file_path, tile_transformation) in enumerate(zip(file_paths, tile_transformations)):

            # Try to memmap the data. If that doesn't work fall back to loading it into memory.
            try:
                data = tifffile.memmap(file_path, mode="r")
            except ValueError:
                print(f"Could not memmap the data from {file_path}. Fall back to load it into memory.")
                data = tifffile.imread(file_path)

            print("Converting channel", channel_id, "tile", tile_id, "from", file_path, "with shape", data.shape)
            if scale_factors is None:
                scale_factors = derive_scale_factors(data.shape)

            pybdv.make_bdv(
                data, out_path,
                downscale_factors=scale_factors, downscale_mode="mean",
                n_threads=n_threads,
                resolution=resolution, unit=unit,
                attributes={
                    "channel": {"id": channel_id, "name": channel_name}, "tile": {"id": tile_id},
                    "angle": {"id": 0}, "illumination": {"id": 0}
                },
                affine=tile_transformation,
            )
