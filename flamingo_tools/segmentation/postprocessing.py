import multiprocessing as mp
from concurrent import futures
from typing import Callable, Tuple, Optional

import elf.parallel as parallel
import numpy as np
import nifty.tools as nt
import pandas as pd
import vigra

from elf.io import open_file
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree, ConvexHull
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


#
# Spatial statistics:
# Three different spatial statistics implementations that
# can be used as the basis of a filtering criterion.
#


def nearest_neighbor_distance(table: pd.DataFrame, n_neighbors: int = 10) -> np.ndarray:
    """Compute the average distance to the n nearest neighbors.

    Args:
        table: The table with the centroid coordinates.
        n_neighbors: The number of neighbors to take into account for the distance computation.

    Returns:
        The average distances to the n nearest neighbors.
    """
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    centroids = np.array(centroids)

    # Nearest neighbor is always itself, so n_neighbors+=1.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(centroids)
    distances, indices = nbrs.kneighbors(centroids)

    # Average distance to nearest neighbors
    distance_avg = np.array([sum(d) / len(d) for d in distances[:, 1:]])
    return distance_avg


def local_ripleys_k(table: pd.DataFrame, radius: float = 15, volume: Optional[float] = None) -> np.ndarray:
    """Compute the local Ripley's K function for each point in a 2D / 3D.

    Args:
        table: The table with the centroid coordinates.
        radius: The radius within which to count neighboring points.
        volume: The area (2D) or volume (3D) of the study region. If None, it is estimated from the convex hull.

    Returns:
        An array containing the local K values for each point.
    """
    points = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    points = np.array(points)
    n_points, dim = points.shape

    if dim not in (2, 3):
        raise ValueError("Points array must be of shape (n_points, 2) or (n_points, 3).")

    # Estimate area/volume if not provided.
    if volume is None:
        hull = ConvexHull(points)
        volume = hull.volume  # For 2D, 'volume' is area; for 3D, it's volume.

    # Compute point density.
    density = n_points / volume

    # Build a KD-tree for efficient neighbor search.
    tree = cKDTree(points)

    # Count neighbors within the specified radius for each point
    counts = tree.query_ball_point(points, r=radius)
    local_counts = np.array([len(c) - 1 for c in counts])  # Exclude the point itself

    # Normalize by density to get local K values
    local_k = local_counts / density
    return local_k


def neighbors_in_radius(table: pd.DataFrame, radius: float = 15) -> np.ndarray:
    """Compute the number of neighbors within a given radius.

    Args:
        table: The table with the centroid coordinates.
        radius: The radius within which to count neighboring points.

    Returns:
        An array containing the number of neighbors within the given radius.
    """
    points = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    points = np.array(points)

    dist_matrix = distance.pdist(points)
    dist_matrix = distance.squareform(dist_matrix)

    # Create sparse matrix of connections within the threshold distance.
    sparse_matrix = csr_matrix(dist_matrix < radius, dtype=int)

    # Sum each row to count neighbors.
    neighbor_counts = sparse_matrix.sum(axis=1)
    return np.array(neighbor_counts)


#
# Filtering function:
# Filter the segmentation based on a spatial statistics from above.
#

# FIXME: functions causes ValueError by using arrays of different lengths


def _compute_table(segmentation):
    segmentation, n_ids, _ = vigra.analysis.relabelConsecutive(segmentation[:], start_label=1, keep_zeros=True)
    props = measure.regionprops(segmentation)
    coordinates = np.array([prop.centroid for prop in props])[1:]
    label_ids = np.unique(segmentation)[1:]
    sizes = np.array([prop.area for prop in props])[1:]
    table = pd.DataFrame({
        "label_id": label_ids,
        "n_pixels": sizes,
        "anchor_x": coordinates[:, 2],
        "anchor_y": coordinates[:, 1],
        "anchor_z": coordinates[:, 0],
    })
    return table


def filter_segmentation(
    segmentation: np.typing.ArrayLike,
    output_path: str,
    spatial_statistics: Callable,
    threshold: float,
    min_size: int = 1000,
    table: Optional[pd.DataFrame] = None,
    output_key: str = "segmentation_postprocessed",
    **spatial_statistics_kwargs,
) -> Tuple[int, int]:
    """Postprocessing step to filter isolated objects from a segmentation.

    Instance segmentations are filtered based on spatial statistics and a threshold.
    In addition, objects smaller than a given size are filtered out.

    Args:
        segmentation: Dataset containing the segmentation
        output_path: Output path for postprocessed segmentation
        spatial_statistics: Function to calculate density measure for elements of segmentation
        threshold: Distance in micrometer to check for neighbors
        min_size: Minimal number of pixels for filtering small instances
        table: Dataframe of segmentation table
        output_key: Output key for postprocessed segmentation
        spatial_statistics_kwargs: Arguments for spatial statistics function

    Returns:
        n_ids
        n_ids_filtered
    """
    # Compute the table on the fly.
    # NOTE: this currently doesn't work for large segmentations.
    if table is None:
        table = _compute_table(segmentation)
    n_ids = len(table)

    # First apply the size filter.
    table = table[table.n_pixels > min_size]
    stat_values = spatial_statistics(table, **spatial_statistics_kwargs)

    keep_mask = np.array(stat_values > threshold).squeeze()
    keep_ids = table.label_id.values[keep_mask]

    shape = segmentation.shape
    block_shape = (128, 128, 128)
    chunks = (128, 128, 128)

    blocking = nt.blocking([0] * len(shape), shape, block_shape)

    output = open_file(output_path, mode="a")
    output_dataset = output.create_dataset(
        output_key, shape=shape, dtype=segmentation.dtype,
        chunks=chunks, compression="gzip"
    )

    def filter_chunk(block_id):
        """Set all points within a chunk to zero if they match filter IDs.
        """
        block = blocking.getBlock(block_id)
        volume_index = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data = segmentation[volume_index]
        data[np.isin(data, keep_ids)] = 0
        output_dataset[volume_index] = data

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as filter_pool:
        list(tqdm(filter_pool.map(filter_chunk, range(blocking.numberOfBlocks)), total=blocking.numberOfBlocks))

    seg_filtered, n_ids_filtered, _ = parallel.relabel_consecutive(
        output_dataset, start_label=1, keep_zeros=True, block_shape=block_shape
    )

    return n_ids, n_ids_filtered
