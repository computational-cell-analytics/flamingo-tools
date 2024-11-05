import multiprocessing as mp
import os
import re
import xml.etree.ElementTree as ET

from glob import glob
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pybdv
import tifffile

from cluster_tools.utils.volume_utils import write_format_metadata
from elf.io import open_file
from skimage.transform import rescale


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

    unit = "micrometer"

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


def read_metadata_flamingo(metadata_path, offset=None, parse_affine=False):
    resolution, unit = None, None

    resolution, unit = _read_resolution_and_unit_flamingo(metadata_path)
    start_position = _read_start_position_flamingo(metadata_path)

    def _pos_to_trafo(pos):
        if offset is not None:
            pos -= offset

        # NOTE: the scale should be kept at 1.
        # This is only here for development purposes,
        # to support handling downsampled datasets.
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

    if parse_affine:
        transformation = _pos_to_trafo(start_position)
    else:
        transformation = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
    # We have to reverse the resolution because pybdv expects ZYX.
    return resolution[::-1], unit, transformation


# TODO derive the scale factors from the shape rather than hard-coding it to 5 levels
def derive_scale_factors(shape):
    scale_factors = [[2, 2, 2]] * 5
    return scale_factors


def _to_ome_zarr(data, out_path, scale_factors, timepoint, setup_id, attributes, unit, resolution):
    n_threads = mp.cpu_count()
    chunks = (128, 128, 128)

    # Write the base dataset.
    base_key = f"setup{setup_id}/timepoint{timepoint}"

    with open_file(out_path, "a") as f:
        ds = f.create_dataset(f"{base_key}/s0", shape=data.shape, compression="gzip",
                              chunks=chunks, dtype=data.dtype)
        ds.n_threads = n_threads
        ds[:] = data

        # TODO parallelized implementation.
        # Do downscaling.
        for level, scale_factor in enumerate(scale_factors, 1):
            inv_scale = [1.0 / sc for sc in scale_factor]
            data = rescale(data, inv_scale, preserve_range=True).astype(data.dtype)
            ds = f.create_dataset(f"{base_key}/s{level}", shape=data.shape, compression="gzip",
                                  chunks=chunks, dtype=data.dtype)
            ds.n_threads = n_threads
            ds[:] = data

        g = f[f"setup{setup_id}"]
        g.attrs.update(attributes)

    # Write the ome zarr metadata.
    metadata_dict = {"unit": unit, "resolution": resolution}
    write_format_metadata(
        "ome.zarr", out_path, metadata_dict, scale_factors=scale_factors, prefix=base_key
    )


def flamingo_filename_parser(file_path, name_mapping):
    filename = os.path.basename(file_path)

    # Extract the timepoint.
    match = re.search(r'_t(\d+)_', filename)
    if match:
        timepoint = int(match.group(1))
    else:
        timepoint = 0

    # Extract the additional attributes.
    attributes = {}
    if name_mapping is None:
        name_mapping = {}

    # Extract the channel.
    match = re.search(r'_C(\d+)_', filename)
    channel = int(match.group(1)) if match else 0
    channel_mapping = name_mapping.get("channel", {})
    attributes["channel"] = {"id": channel, "name": channel_mapping.get(channel, str(channel))}

    # Extract the tile.
    match = re.search(r'_R(\d+)_', filename)
    tile = int(match.group(1)) if match else 0
    tile_mapping = name_mapping.get("tile", {})
    attributes["tile"] = {"id": tile, "name": tile_mapping.get(tile, str(tile))}

    # Extract the illumination.
    match = re.search(r'_I(\d+)_', filename)
    illumination = int(match.group(1)) if match else 0
    illumination_mapping = name_mapping.get("illumination", {})
    attributes["illumination"] = {"id": illumination, "name": illumination_mapping.get(illumination, str(illumination))}

    # Extract D. TODO what is this?
    match = re.search(r'_D(\d+)_', filename)
    D = int(match.group(1)) if match else 0
    D_mapping = name_mapping.get("D", {})
    attributes["D"] = {"id": D, "name": D_mapping.get(D, str(D))}

    # BDV also supports an angle attribute, but it does not seem to be stored in the filename
    # "angle": {"id": 0, "name": "0"}

    attribute_id = f"c{channel}-t{tile}-i{illumination}-d{D}"
    return timepoint, attributes, attribute_id


def _write_missing_views(out_path):
    xml_path = Path(out_path).with_suffix(".xml")
    assert os.path.exists(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    seqdesc = root.find("SequenceDescription")
    ET.SubElement(seqdesc, "MissingViews")

    pybdv.metadata.indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


def _parse_shape(metadata_file):
    depth, height, width = None, None, None

    with open(metadata_file, "r") as f:
        for line in f.readlines():
            line = line.strip().rstrip("\n")
            if line.startswith("AOI width"):
                width = int(line.split(" ")[-1])
            if line.startswith("AOI height"):
                height = int(line.split(" ")[-1])
            if line.startswith("Number of planes saved"):
                depth = int(line.split(" ")[-1])

    assert depth is not None
    assert height is not None
    assert width is not None
    return (depth, height, width)


def _load_data(file_path, metadata_file):
    if Path(file_path).suffix == ".raw":
        shape = _parse_shape(metadata_file)
        data = np.memmap(file_path, mode="r", dtype="uint16", shape=shape)
    else:
        try:
            data = tifffile.memmap(file_path, mode="r")
        except ValueError:
            print(f"Could not memmap the data from {file_path}. Fall back to load it into memory.")
            data = tifffile.imread(file_path)
    return data


def convert_lightsheet_to_bdv(
    root: str,
    out_path: str,
    file_ext: str = ".tif",
    attribute_parser: callable = flamingo_filename_parser,
    attribute_names: Optional[Dict[str, Dict[int, str]]] = None,
    metadata_file_name_pattern: Optional[str] = None,
    metadata_root: Optional[str] = None,
    metadata_type: str = "flamingo",
    center_tiles: bool = False,
    resolution: Optional[List[float]] = None,
    unit: Optional[str] = None,
    scale_factors: Optional[List[List[int]]] = None,
    n_threads: Optional[int] = None,
) -> None:
    """Convert channels and tiles acquired with a lightsheet microscope.

    The data is converted to the bdv-n5 file format and can be opened with BigDataViewer
    or BigStitcher. This function is written with data layout and metadata of flamingo
    microscopes in mind, but could potentially be adapted to other data formats.

    TODO explain the attribute parsing.

    Args:
        root: Folder that contains the image data stored as tifs.
            This function will take into account all tif files in folders beneath this root directory.
        out_path: Output path where the converted data is saved.
        file_ext: The name of the file extension. By default assumes tif files (.tif).
            Change to '.raw' to read files stored in raw format instead.
        attribute_parser: TODO
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
    convert_to_ome_zarr = False
    if ext == "":
        out_path = str(Path(out_path).with_suffix(".n5"))
    elif ext == ".zarr":
        convert_to_ome_zarr = True

    files = sorted(glob(os.path.join(root, f"**/*{file_ext}"), recursive=True))
    # Raise an error if we could not find any files.
    if len(files) == 0:
        raise ValueError(f"Could not find any files in {root} with extension {file_ext}.")

    if metadata_file_name_pattern is None:
        metadata_files = [None] * len(files)
        offset = None
    else:
        metadata_files = sorted(
            glob(
                os.path.join(root if metadata_root is None else metadata_root, f"**/{metadata_file_name_pattern}"),
                recursive=True
            )
        )
        assert len(metadata_files) == len(files), f"{len(metadata_files)}, {len(files)}"

        if center_tiles:
            start_positions = []
            for mpath in metadata_files:
                start_positions.append(_read_start_position_flamingo(mpath))
            offset = np.min(start_positions, axis=0)
        else:
            offset = None

    next_setup_id = 0
    attrs_to_setups = {}

    for file_path, metadata_file in zip(files, metadata_files):
        timepoint, attributes, aid = attribute_parser(file_path, attribute_names)

        if aid in attrs_to_setups:
            setup_id = attrs_to_setups[aid]
        else:
            attrs_to_setups[aid] = next_setup_id
            setup_id = next_setup_id
            next_setup_id += 1

        # Read the metadata if it was given.
        if metadata_file is None:  # No metadata given.
            # We don't use any tile transformation.
            tile_transformation = None
            # Set resolution and unit to their default values if they were not passed.
            if resolution is None:
                resolution = [1.0, 1.0, 1.0]
            if unit is None:
                unit = "pixel"

        else:  # We have metadata and read it.
            # NOTE: we don't add the calibration transformation here, as this
            # leads to issues with the BigStitcher export.
            resolution, unit, tile_transformation = read_metadata_flamingo(
                metadata_file, offset, parse_affine=False
            )

        print(f"Converting tp={timepoint}, channel={attributes['channel']}, tile={attributes['tile']}")
        data = _load_data(file_path, metadata_file)
        if scale_factors is None:
            scale_factors = derive_scale_factors(data.shape)

        if convert_to_ome_zarr:
            _to_ome_zarr(data, out_path, scale_factors, timepoint, setup_id, attributes, unit, resolution)
        else:
            pybdv.make_bdv(
                data, out_path,
                downscale_factors=scale_factors, downscale_mode="mean",
                n_threads=n_threads,
                resolution=resolution, unit=unit,
                attributes=attributes,
                affine=tile_transformation,
                timepoint=timepoint,
                setup_id=setup_id,
                chunks=(128, 128, 128),
            )

    # We don't need to add additional xml metadata if we convert to ome-zarr.
    if convert_to_ome_zarr:
        return

    # Add an empty missing views field.
    # This is expected by BigStitcher.
    _write_missing_views(out_path)


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
