import multiprocessing as mp
import os

from glob import glob
from pathlib import Path
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

    # NOTE: The resolution for the flamingo system is isotropic.
    # So we can just return the plane spacing value to get it.
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
        # scale = 4
        scale = 1

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


# TODO derive the scale factors from the shape rather than hard-coding it to 5 levels
def derive_scale_factors(shape):
    scale_factors = [[2, 2, 2]] * 5
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
    """Convert channels and tiles acquired with a lightsheet microscope.

    The data is converted to the bdv-n5 file format and can be opened with BigDataViewer
    or BigStitcher. This function is written with data layout and metadata of flamingo
    microscopes in mind, but could potentially be adapted to other data formats.
    We currently don't support multiple timepoints, but support can be added if needed.

    This function assumes the following input data format:
    <ROOT>/<CHANNEL1>/<TILE1>.tif
                     /<TILE2>.tif
                     /...
          /<CHANNEL2>/<TILE1>.tif
                     /<TILE2>.tif
                     /...

    Args:
        root: Folder that contains the folders with tifs for each channel.
        channel_folders: Dictionary that maps the name of each channel to the corresponding folder name
            underneath the root folder.
        image_file_name_pattern: The pattern for the names of the tifs that contain the data.
            This expects a glob pattern (name with '*') to select the corresponding tif files .
            The simplest pattern that should work in most cases is '*.tif'.
        out_path: Output path where the converted data is saved.
        metadata_file_name_pattern: The pattern for the names of files that contain the metadata.
            For flamingo metadata the following pattern should work: '*_Settings.txt'.
        metadata_root: Different root folder for the metadata. By default 'root' is used here as well.
        metadata_type: The type of the metadata (for now only 'flamingo' is supported).
        center_tiles: Whether to move the tiles to the origin.
        resolution: The physical size of one pixel. This is only used if the metadata is not read from file.
        unit: The unit of the given resolution. This is only used if the metadata is not read from file.
        scale_factors: The scale factors for downsampling the image data.
            By default sensible factors will be determined based on the shape of the data.
            If you want to set the scale factors manually then you have to pass them as a list with the
            downsampling factors for each level. E.g.:
            - [[2, 2, 2], [2, 2, 2]] to downsample isotropically by a factor of 2 for two times.
            - [[1, 2, 2], [1, 2, 2]] to downsample anisotropically for two times.
            - [[1, 2, 2], [2, 2, 2]] to downsample anisotroically once and then isotropically.
        n_threads: The number of threads to use for parallelizing the data conversion.
            By default all available CPU cores will be used.
    """
    if metadata_type != "flamingo":
        raise ValueError(f"Invalid metadata type: {metadata_type}.")
    if n_threads is None:
        n_threads = mp.cpu_count()

    # Make sure we convert to n5, in case no extension is passed.
    ext = os.path.splitext(out_path)[1]
    if ext == "":
        out_path = str(Path(out_path).with_suffix(".n5"))

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

        if channel_name is None or channel_name.strip() == "": #channel name is empty, assign channel id as name
            channel_name = str(channel_id)

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
                    "channel": {"id": channel_id, "name": channel_name}, "tile": {"id": tile_id, "name": str(tile_id)},
                    "angle": {"id": 0, "name": "0"}, "illumination": {"id": 0, "name": "0"}
                },
                affine=tile_transformation,
            )


# TODO expose more arguments via CLI.
def convert_lightsheet_to_bdv_cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert lightsheet data to format compatible with BigDataViewer / BigStitcher. "
                    "Example useage: To convert the synthetic data created via create_synthetic_data.py run: \n"
                    "python convert_flamingo_data.py -i synthetic_data -c channel0 channel1 -f *.tif -o synthetic.n5"
    )
    parser.add_argument(
        "--input_root", "-i", required=True,
        help="Folder that contains the folders with tifs for each channel."
    )
    parser.add_argument(
        "--channel_folders", "-c", nargs="+", required=True,
        help="Name of folders with the data for each channel."
    )
    parser.add_argument(
        "--image_file_name_pattern", "-f", required=True,
        help="The pattern for the names of the tifs that contain the data. "
             "This expects a glob pattern (name with '*') to select the corresponding tif files."
             "The simplest pattern that should work in most cases is '*.tif'."
    )
    parser.add_argument(
        "--out_path", "-o", required=True,
        help="Output path where the converted data is saved."
    )

    args = parser.parse_args()
    channel_folders = {name: name for name in args.channel_folders}
    convert_lightsheet_to_bdv(
        args.input_root, channel_folders, args.image_file_name_pattern, args.out_path,
    )
