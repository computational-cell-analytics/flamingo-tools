import multiprocessing as mp
from concurrent import futures
from typing import Optional

import numpy as np
import pandas as pd
import trimesh
from skimage.measure import marching_cubes
from tqdm import tqdm

from .file_utils import read_image_data


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


# Could also support s3 directly?
def compute_object_measures(
    image_path: str,
    segmentation_path: str,
    segmentation_table_path: str,
    output_table_path: str,
    image_key: Optional[str] = None,
    segmentation_key: Optional[str] = None,
    n_threads: Optional[int] = None,
    resolution: float = 0.38,
):
    """

    Args:
        image_path:
        segmentation_path:
        segmentation_table_path:
        output_table_path:
        image_key:
        segmentation_key:
        n_threads:
        resolution:
    """
    # First, we load the pre-computed segmentation table from MoBIE.
    table = pd.read_csv(segmentation_table_path, sep="\t")

    # Then, open the volumes.
    image = read_image_data(image_path, image_key)
    segmentation = read_image_data(segmentation_path, segmentation_key)

    def intensity_measures(seg_id):
        # Get the bounding box.
        row = table[table.label_id == seg_id]

        bb_min = np.array([
            row.bb_min_z.item(), row.bb_min_y.item(), row.bb_min_x.item()
        ]) / resolution
        bb_min = np.round(bb_min, 0).astype("uint32")

        bb_max = np.array([
            row.bb_max_z.item(), row.bb_max_y.item(), row.bb_max_x.item()
        ]) / resolution
        bb_max = np.round(bb_max, 0).astype("uint32")

        bb = tuple(
            slice(max(bmin - 1, 0), min(bmax + 1, sh))
            for bmin, bmax, sh in zip(bb_min, bb_max, image.shape)
        )

        local_image = image[bb]
        mask = segmentation[bb] == seg_id
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

    seg_ids = table.label_id.values
    n_threads = mp.cpu_count() if n_threads is None else n_threads
    with futures.ThreadPoolExecutor(n_threads) as pool:
        measures = list(tqdm(
            pool.map(intensity_measures, seg_ids),
            total=len(seg_ids), desc="Compute intensity measures"
        ))

    # Create the result table and save it.
    keys = measures[0].keys()
    measures = pd.DataFrame({k: [measure[k] for measure in measures] for k in keys})
    measures.to_csv(output_table_path, sep="\t", index=False)
