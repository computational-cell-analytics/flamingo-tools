import multiprocessing as mp
import os
from concurrent import futures
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import trimesh
from skimage.measure import marching_cubes, regionprops_table
from tqdm import tqdm

from .file_utils import read_image_data
from .segmentation.postprocessing import compute_table_on_the_fly


def _measure_volume_and_surface(mask, resolution):
    # Use marching_cubes for 3D data
    verts, faces, normals, _ = marching_cubes(mask, spacing=(resolution,) * 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    surface = mesh.area
    if mesh.is_watertight:
        volume = np.abs(mesh.volume)
    else:
        volume = np.nan

    return volume, surface


def _get_bounding_box(table, seg_id, resolution, shape):
    row = table[table.label_id == seg_id]

    bb_min = np.array([
        row.bb_min_z.item(), row.bb_min_y.item(), row.bb_min_x.item()
    ]).astype("float32") / resolution
    bb_min = np.round(bb_min, 0).astype("int32")

    bb_max = np.array([
        row.bb_max_z.item(), row.bb_max_y.item(), row.bb_max_x.item()
    ]).astype("float32") / resolution
    bb_max = np.round(bb_max, 0).astype("int32")

    bb = tuple(
        slice(max(bmin - 1, 0), min(bmax + 1, sh))
        for bmin, bmax, sh in zip(bb_min, bb_max, shape)
    )
    return bb


def _default_object_features(seg_id, table, image, segmentation, resolution):
    bb = _get_bounding_box(table, seg_id, resolution, image.shape)

    local_image = image[bb]
    mask = segmentation[bb] == seg_id
    assert mask.sum() > 0, f"Segmentation ID {seg_id} is empty."
    masked_intensity = local_image[mask]

    # Do the base intensity measurements.
    measures = {
        "label_id": seg_id,
        "mean": np.mean(masked_intensity),
        "stdev": np.std(masked_intensity),
        "min": np.min(masked_intensity),
        "max": np.max(masked_intensity),
        "median": np.median(masked_intensity),
    }
    for percentile in (5, 10, 25, 75, 90, 95):
        measures[f"percentile-{percentile}"] = np.percentile(masked_intensity, percentile)

    # Do the volume and surface measurement.
    volume, surface = _measure_volume_and_surface(mask, resolution)
    measures["volume"] = volume
    measures["surface"] = surface
    return measures


def _regionprops_features(seg_id, table, image, segmentation, resolution):
    bb = _get_bounding_box(table, seg_id, resolution, image.shape)

    local_image = image[bb]
    local_segmentation = segmentation[bb]
    mask = local_segmentation == seg_id
    assert mask.sum() > 0, f"Segmentation ID {seg_id} is empty."
    local_segmentation[~mask] = 0

    features = regionprops_table(
        local_segmentation, local_image, properties=[
            "label", "area", "axis_major_length", "axis_minor_length",
            "equivalent_diameter_area", "euler_number", "extent",
            "feret_diameter_max", "inertia_tensor_eigvals",
            "intensity_max", "intensity_mean", "intensity_min",
            "intensity_std", "moments_central",
            "moments_weighted", "solidity",
        ]
    )

    features["label_id"] = features.pop("label")
    return features


# Maybe also support:
# - spherical harmonics
# - line profiles
FEATURE_FUNCTIONS = {
    "default": _default_object_features,
    "skimage": _regionprops_features,
}
"""The different feature functions that are supported in `compute_object_measures` and
that can be selected via the feature_set argument. Currently this supports:
- 'default': The default features which compute standard intensity statistics and volume + surface.
- 'skimage': The scikit image regionprops features.
"""


def compute_object_measures_impl(
    image: np.typing.ArrayLike,
    segmentation: np.typing.ArrayLike,
    n_threads: Optional[int] = None,
    resolution: float = 0.38,
    table: Optional[pd.DataFrame] = None,
    feature_set: str = "default",
) -> pd.DataFrame:
    """Compute simple intensity and morphology measures for each segmented cell in a segmentation.

    See `compute_object_measures` for details.

    Args:
        image: The image data.
        segmentation: The segmentation.
        n_threads: The number of threads to use for computation.
        resolution: The resolution / voxel size of the data.
        table: The segmentation table. Will be computed on the fly if it is not given.
        feature_set: The features to compute for each object. Refer to `FEATURE_FUNCTIONS` for details.

    Returns:
        The table with per object measurements.
    """
    if table is None:
        table = compute_table_on_the_fly(segmentation, resolution=resolution)

    if feature_set not in FEATURE_FUNCTIONS:
        raise ValueError
    measure_function = partial(
        FEATURE_FUNCTIONS[feature_set],
        table=table,
        image=image,
        segmentation=segmentation,
        resolution=resolution
    )

    seg_ids = table.label_id.values
    assert len(seg_ids) > 0, "The segmentation table is empty."
    n_threads = mp.cpu_count() if n_threads is None else n_threads
    with futures.ThreadPoolExecutor(n_threads) as pool:
        measures = list(tqdm(
            pool.map(measure_function, seg_ids), total=len(seg_ids), desc="Compute intensity measures"
        ))

    # Create the result table and save it.
    keys = measures[0].keys()
    measures = pd.DataFrame({k: [measure[k] for measure in measures] for k in keys})
    return measures


# Could also support s3 directly?
def compute_object_measures(
    image_path: str,
    segmentation_path: str,
    segmentation_table_path: Optional[str],
    output_table_path: str,
    image_key: Optional[str] = None,
    segmentation_key: Optional[str] = None,
    n_threads: Optional[int] = None,
    resolution: float = 0.38,
    force: bool = False,
    feature_set: str = "default",
) -> None:
    """Compute simple intensity and morphology measures for each segmented cell in a segmentation.

    By default, this computes the mean, standard deviation, minimum, maximum, median and
    5th, 10th, 25th, 75th, 90th and 95th percentile of the intensity image
    per cell, as well as the volume and surface.
    Other measurements can be computed by changing the feature_set argument.

    Args:
        image_path: The filepath to the image data. Either a tif or hdf5/zarr/n5 file.
        segmentation_path: The filepath to the segmentation data. Either a tif or hdf5/zarr/n5 file.
        segmentation_table_path: The path to the segmentation table in MoBIE format.
        output_table_path: The path for saving the segmentation with intensity measures.
        image_key: The key (= internal path) for the image data. Not needed fir tif.
        segmentation_key: The key (= internal path) for the segmentation data. Not needed for tif.
        n_threads: The number of threads to use for computation.
        resolution: The resolution / voxel size of the data.
        force: Whether to overwrite an existing output table.
        feature_set: The features to compute for each object. Refer to `FEATURE_FUNCTIONS` for details.
    """
    if os.path.exists(output_table_path) and not force:
        return

    # First, we load the pre-computed segmentation table from MoBIE.
    if segmentation_table_path is None:
        table = None
    else:
        table = pd.read_csv(segmentation_table_path, sep="\t")

    # Then, open the volumes.
    image = read_image_data(image_path, image_key)
    segmentation = read_image_data(segmentation_path, segmentation_key)

    measures = compute_object_measures_impl(
        image, segmentation, n_threads, resolution, table=table, feature_set=feature_set,
    )
    measures.to_csv(output_table_path, sep="\t", index=False)
