import math
import multiprocessing as mp
from concurrent import futures
from typing import Callable, List, Optional, Tuple

import elf.parallel as parallel
import numpy as np
import nifty.tools as nt
import networkx as nx
import pandas as pd

from elf.io import open_file
from scipy.sparse import csr_matrix
from scipy.spatial import distance
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


def _compute_table(segmentation, resolution):
    props = measure.regionprops(segmentation)
    label_ids = np.array([prop.label for prop in props])
    coordinates = np.array([prop.centroid for prop in props])
    # transform pixel distance to physical units
    coordinates = coordinates * resolution
    sizes = np.array([prop.area for prop in props])
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
    resolution: float = 0.38,
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
        resolution: Resolution of segmentation in micrometer
        output_key: Output key for postprocessed segmentation
        spatial_statistics_kwargs: Arguments for spatial statistics function

    Returns:
        n_ids
        n_ids_filtered
    """
    # Compute the table on the fly.
    # NOTE: this currently doesn't work for large segmentations.
    if table is None:
        table = _compute_table(segmentation, resolution=resolution)
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


def erode_subset(
    table: pd.DataFrame,
    iterations: int = 1,
    min_cells: Optional[int] = None,
    threshold: int = 35,
    keyword: str = "distance_nn100",
) -> pd.DataFrame:
    """Erode coordinates of dataframe according to a keyword and a threshold.
    Use a copy of the dataframe as an input, if it should not be edited.

    Args:
        table: Dataframe of segmentation table.
        iterations: Number of steps for erosion process.
        min_cells: Minimal number of rows. The erosion is stopped after falling below this limit.
        threshold: Upper threshold for removing elements according to the given keyword.
        keyword: Keyword of dataframe for erosion.

    Returns:
        The dataframe containing elements left after the erosion.
    """
    print("initial length", len(table))
    n_neighbors = 100
    for i in range(iterations):
        table = table[table[keyword] < threshold]

        distance_avg = nearest_neighbor_distance(table, n_neighbors=n_neighbors)

        if min_cells is not None and len(distance_avg) < min_cells:
            print(f"{i}-th iteration, length of subset {len(table)}, stopping erosion")
            break

        table.loc[:, 'distance_nn'+str(n_neighbors)] = list(distance_avg)

        print(f"{i}-th iteration, length of subset {len(table)}")

    return table


def downscaled_centroids(
    table: pd.DataFrame,
    scale_factor: int,
    ref_dimensions: Optional[Tuple[float, float, float]] = None,
    downsample_mode: str = "accumulated",
) -> np.typing.NDArray:
    """Downscale centroids in dataframe.

    Args:
        table: Dataframe of segmentation table.
        scale_factor: Factor for downscaling coordinates.
        ref_dimensions: Reference dimensions for downscaling. Taken from centroids if not supplied.
        downsample_mode: Flag for downsampling, either 'accumulated', 'capped', or 'components'.

    Returns:
        The downscaled array
    """
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    centroids_scaled = [(c[0] / scale_factor, c[1] / scale_factor, c[2] / scale_factor) for c in centroids]

    if ref_dimensions is None:
        bounding_dimensions = (max(table["anchor_x"]), max(table["anchor_y"]), max(table["anchor_z"]))
        bounding_dimensions_scaled = tuple([round(b // scale_factor + 1) for b in bounding_dimensions])
        new_array = np.zeros(bounding_dimensions_scaled)

    else:
        bounding_dimensions_scaled = tuple([round(b // scale_factor + 1) for b in ref_dimensions])
        new_array = np.zeros(bounding_dimensions_scaled)

    if downsample_mode == "accumulated":
        for c in centroids_scaled:
            new_array[int(c[0]), int(c[1]), int(c[2])] += 1

    elif downsample_mode == "capped":
        for c in centroids_scaled:
            new_array[int(c[0]), int(c[1]), int(c[2])] = 1

    elif downsample_mode == "components":
        if "component_labels" not in table.columns:
            raise KeyError("Dataframe must continue key 'component_labels' for downsampling with mode 'components'.")
        component_labels = list(table["component_labels"])
        for comp, centr in zip(component_labels, centroids_scaled):
            if comp != 0:
                new_array[int(centr[0]), int(centr[1]), int(centr[2])] = comp

    else:
        raise ValueError("Choose one of the downsampling modes 'accumulated', 'capped', or 'components'.")

    new_array = np.round(new_array).astype(int)

    return new_array


def components_sgn(
    table: pd.DataFrame,
    keyword: str = "distance_nn100",
    threshold_erode: Optional[float] = None,
    postprocess_graph: bool = False,
    min_component_length: int = 50,
    min_edge_distance: float = 30,
    iterations_erode: Optional[int] = None,
) -> List[List[int]]:
    """Eroding the SGN segmentation.

    Args:
        table: Dataframe of segmentation table.
        keyword: Keyword of the dataframe column for erosion.
        threshold_erode: Threshold of column value after erosion step with spatial statistics.
        postprocess_graph: Post-process graph connected components by searching for near points.
        min_component_length: Minimal length for filtering out connected components.
        min_edge_distance: Minimal distance in micrometer between points to create edges for connected components.
        iterations_erode: Number of iterations for erosion, normally determined automatically.

    Returns:
        Subgraph components as lists of label_ids of dataframe.
    """
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    labels = [int(i) for i in list(table["label_id"])]

    distance_nn = list(table[keyword])
    distance_nn.sort()

    if len(table) < 20000:
        iterations = iterations_erode if iterations_erode is not None else 0
        min_cells = None
        average_dist = int(distance_nn[int(len(table) * 0.8)])
        threshold = threshold_erode if threshold_erode is not None else average_dist
    else:
        iterations = iterations_erode if iterations_erode is not None else 15
        min_cells = 20000
        threshold = threshold_erode if threshold_erode is not None else 40

    print(f"Using threshold of {threshold} micrometer for eroding segmentation with keyword {keyword}.")

    new_subset = erode_subset(table.copy(), iterations=iterations,
                              threshold=threshold, min_cells=min_cells, keyword=keyword)

    # create graph from coordinates of eroded subset
    centroids_subset = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    labels_subset = [int(i) for i in list(new_subset["label_id"])]
    coords = {}
    for index, element in zip(labels_subset, centroids_subset):
        coords[index] = element

    graph = nx.Graph()
    for num, pos in coords.items():
        graph.add_node(num, pos=pos)

    # create edges between points whose distance is less than threshold min_edge_distance
    for i in coords:
        for j in coords:
            if i < j:
                dist = math.dist(coords[i], coords[j])
                if dist <= min_edge_distance:
                    graph.add_edge(i, j, weight=dist)

    components = list(nx.connected_components(graph))

    # remove connected components with less nodes than threshold min_component_length
    for component in components:
        if len(component) < min_component_length:
            for c in component:
                graph.remove_node(c)

    components = [list(s) for s in nx.connected_components(graph)]

    # add original coordinates closer to eroded component than threshold
    if postprocess_graph:
        threshold = 15
        for label_id, centr in zip(labels, centroids):
            if label_id not in labels_subset:
                add_coord = []
                for comp_index, component in enumerate(components):
                    for comp_label in component:
                        dist = math.dist(centr, centroids[comp_label - 1])
                        if dist <= threshold:
                            add_coord.append([comp_index, label_id])
                            break
                if len(add_coord) != 0:
                    components[add_coord[0][0]].append(add_coord[0][1])

    return components


def label_components(
    table: pd.DataFrame,
    min_size: int = 1000,
    threshold_erode: Optional[float] = None,
    min_component_length: int = 50,
    min_edge_distance: float = 30,
    iterations_erode: Optional[int] = None,
) -> List[int]:
    """Label components using graph connected components.

    Args:
        table: Dataframe of segmentation table.
        min_size: Minimal number of pixels for filtering small instances.
        threshold_erode: Threshold of column value after erosion step with spatial statistics.
        min_component_length: Minimal length for filtering out connected components.
        min_edge_distance: Minimal distance in micrometer between points to create edges for connected components.
        iterations_erode: Number of iterations for erosion, normally determined automatically.

    Returns:
        List of component label for each point in dataframe. 0 - background, then in descending order of size
    """

    # First, apply the size filter.
    entries_filtered = table[table.n_pixels < min_size]
    table = table[table.n_pixels >= min_size]

    components = components_sgn(table, threshold_erode=threshold_erode, min_component_length=min_component_length,
                                min_edge_distance=min_edge_distance, iterations_erode=iterations_erode)

    # add size-filtered objects to have same initial length
    table = pd.concat([table, entries_filtered], ignore_index=True)
    table.sort_values("label_id")

    length_components = [len(c) for c in components]
    length_components, components = zip(*sorted(zip(length_components, components), reverse=True))

    component_labels = [0 for _ in range(len(table))]
    # be aware of 'label_id' of dataframe starting at 1
    for lab, comp in enumerate(components):
        for comp_index in comp:
            component_labels[comp_index - 1] = lab + 1

    return component_labels


def postprocess_sgn_seg(
    table: pd.DataFrame,
    min_size: int = 1000,
    threshold_erode: Optional[float] = None,
    min_component_length: int = 50,
    min_edge_distance: float = 30,
    iterations_erode: Optional[int] = None,
) -> pd.DataFrame:
    """Postprocessing SGN segmentation of cochlea.

    Args:
        table: Dataframe of segmentation table.
        min_size: Minimal number of pixels for filtering small instances.
        threshold_erode: Threshold of column value after erosion step with spatial statistics.
        min_component_length: Minimal length for filtering out connected components.
        min_edge_distance: Minimal distance in micrometer between points to create edges for connected components.
        iterations_erode: Number of iterations for erosion, normally determined automatically.

    Returns:
        Dataframe with component labels.
    """

    comp_labels = label_components(table, min_size=min_size, threshold_erode=threshold_erode,
                                   min_component_length=min_component_length,
                                   min_edge_distance=min_edge_distance, iterations_erode=iterations_erode)

    table.loc[:, "component_labels"] = comp_labels

    return table
