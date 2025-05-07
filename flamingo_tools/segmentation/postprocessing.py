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
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label
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


# Postprocess segmentation by erosion using the above spatial statistics.
# Currently implemented using downscaling and looking for connected components
# TODO: Change implementation to graph connected components.


def erode_subset(
    table: pd.DataFrame,
    iterations: Optional[int] = 1,
    min_cells: Optional[int] = None,
    threshold: Optional[int] = 35,
    keyword: Optional[str] = "distance_nn100",
) -> pd.DataFrame:
    """Erode coordinates of dataframe according to a keyword and a threshold.
    Use a copy of the dataframe as an input, if it should not be edited.

    Args:
        table: Dataframe of segmentation table.
        iterations: Number of steps for erosion process.
        min_cells: Minimal number of rows. The erosion is stopped before reaching this number.
        threshold: Upper threshold for removing elements according to the given keyword.
        keyword: Keyword of dataframe for erosion.

    Returns:
        The dataframe containing elements left after the erosion.
    """
    print("initial length", len(table))
    n_neighbors = 100
    for i in range(iterations):
        table = table[table[keyword] < threshold]

        # TODO: support other spatial statistics
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
    downsample_mode: Optional[str] = "accumulated",
) -> np.typing.NDArray:
    """Downscale centroids in dataframe.

    Args:
        table: Dataframe of segmentation table.
        scale_factor: Factor for downscaling coordinates.
        ref_dimensions: Reference dimensions for downscaling. Taken from centroids if not supplied.
        downsample_mode: Flag for downsampling, either 'accumulated', 'capped', or 'components'

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


def coordinates_in_downscaled_blocks(
    table: pd.DataFrame,
    down_array: np.typing.NDArray,
    scale_factor: float,
    distance_component: Optional[int] = 0,
) -> List[int]:
    """Checking if coordinates are within the downscaled array.

    Args:
        table: Dataframe of segmentation table.
        down_array: Downscaled array.
        scale_factor: Factor which was used for downscaling.
        distance_component: Distance in downscaled units to which centroids next to downscaled blocks are included.

    Returns:
        A binary list representing whether the dataframe coordinates are within the array.
    """
    # fill holes in down-sampled array
    down_array[down_array > 0] = 1
    down_array = binary_fill_holes(down_array).astype(np.uint8)

    # check if input coordinates are within down-sampled blocks
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    centroids = [np.floor(np.array([c[0]/scale_factor, c[1]/scale_factor, c[2]/scale_factor])) for c in centroids]

    distance_map = distance_transform_edt(down_array == 0)

    centroids_binary = []
    for c in centroids:
        coord = (int(c[0]), int(c[1]), int(c[2]))
        if down_array[coord] != 0:
            centroids_binary.append(1)
        elif distance_map[coord] <= distance_component:
            centroids_binary.append(1)
        else:
            centroids_binary.append(0)

    return centroids_binary


def erode_sgn_seg_graph(
    table: pd.DataFrame,
    keyword: Optional[str] = "distance_nn100",
    threshold_erode: Optional[float] = None,
) -> List[List[int]]:
    """Eroding the SGN segmentation.

    Args:
        table: Dataframe of segmentation table.
        keyword: Keyword of the dataframe column for erosion.
        threshold_erode: Threshold of column value after erosion step with spatial statistics.

    Returns:
        Subgraph components as lists of label_ids of dataframe.
    """
    print("initial length", len(table))
    distance_nn = list(table[keyword])
    distance_nn.sort()

    if len(table) < 20000:
        iterations = 1
        min_cells = None
        average_dist = int(distance_nn[int(len(table) * 0.8)])
        threshold = threshold_erode if threshold_erode is not None else average_dist
    else:
        iterations = 15
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

    # create edges between points whose distance is less than threshold
    threshold = 30
    for i in coords:
        for j in coords:
            if i < j:
                dist = math.dist(coords[i], coords[j])
                if dist <= threshold:
                    graph.add_edge(i, j, weight=dist)

    components = list(nx.connected_components(graph))

    # remove connected components with less nodes than threshold
    min_length = 100
    for component in components:
        if len(component) < min_length:
            for c in component:
                graph.remove_node(c)

    components = list(nx.connected_components(graph))

    return components


def erode_sgn_seg_downscaling(
    table: pd.DataFrame,
    keyword: Optional[str] = "distance_nn100",
    filter_small_components: Optional[int] = None,
    scale_factor: Optional[float] = 20,
    threshold_erode: Optional[float] = None,
) -> Tuple[np.typing.NDArray, np.typing.NDArray]:
    """Eroding the SGN segmentation.

    Args:
        table: Dataframe of segmentation table.
        keyword: Keyword of the dataframe column for erosion.
        filter_small_components: Filter components smaller after n blocks after labeling.
        scale_factor: Scaling for downsampling.
        threshold_erode: Threshold of column value after erosion step with spatial statistics.

    Returns:
        The labeled components of the downscaled, eroded coordinates.
        The larget connected component of the labeled components.
    """
    ref_dimensions = (max(table["anchor_x"]), max(table["anchor_y"]), max(table["anchor_z"]))
    print("initial length", len(table))
    distance_nn = list(table[keyword])
    distance_nn.sort()

    if len(table) < 20000:
        iterations = 1
        min_cells = None
        average_dist = int(distance_nn[int(len(table) * 0.8)])
        threshold = threshold_erode if threshold_erode is not None else average_dist
    else:
        iterations = 15
        min_cells = 20000
        threshold = threshold_erode if threshold_erode is not None else 40

    print(f"Using threshold of {threshold} micrometer for eroding segmentation with keyword {keyword}.")

    new_subset = erode_subset(table.copy(), iterations=iterations,
                              threshold=threshold, min_cells=min_cells, keyword=keyword)

    eroded_arr = downscaled_centroids(new_subset, scale_factor=scale_factor, ref_dimensions=ref_dimensions)

    # Label connected components
    labeled, num_features = label(eroded_arr)

    # Find the largest component
    sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest_label = np.argmax(sizes) + 1

    # Extract only the largest component
    largest_component = (labeled == largest_label).astype(np.uint8)
    largest_component_filtered = binary_fill_holes(largest_component).astype(np.uint8)

    # filter small sizes
    if filter_small_components is not None:
        for (size, feature) in zip(sizes, range(1, num_features + 1)):
            if size < filter_small_components:
                labeled[labeled == feature] = 0

    return labeled, largest_component_filtered


def get_components(
    table: pd.DataFrame,
    labeled: np.typing.NDArray,
    scale_factor: float,
    distance_component: Optional[int] = 0,
) -> List[int]:
    """Indexing coordinates according to labeled array.

    Args:
        table: Dataframe of segmentation table.
        labeled: Array containing differently labeled components.
        scale_factor: Scaling for downsampling.
        distance_component: Distance in downscaled units to which centroids next to downscaled blocks are included.

    Returns:
        List of component labels.
    """
    unique_labels = list(np.unique(labeled))

    # sort non-background labels according to size, descending
    unique_labels = [i for i in unique_labels if i != 0]
    sizes = [(labeled == i).sum() for i in unique_labels]
    sizes, unique_labels = zip(*sorted(zip(sizes, unique_labels), reverse=True))

    component_labels = [0 for _ in range(len(table))]
    for label_index, l in enumerate(unique_labels):
        label_arr = (labeled == l).astype(np.uint8)
        centroids_binary = coordinates_in_downscaled_blocks(table, label_arr,
                                                            scale_factor, distance_component=distance_component)
        for num, c in enumerate(centroids_binary):
            if c != 0:
                component_labels[num] = label_index + 1

    return component_labels


def component_labels_graph(table: pd.DataFrame) -> List[int]:
    """Label components using graph connected components.

    Args:
        table: Dataframe of segmentation table.

    Returns:
        List of component label for each point in dataframe.
    """
    components = erode_sgn_seg_graph(table)

    length_components = [len(c) for c in components]
    length_components, components = zip(*sorted(zip(length_components, components), reverse=True))

    component_labels = [0 for _ in range(len(table))]
    for lab, comp in enumerate(components):
        for comp_index in comp:
            component_labels[comp_index - 1] = lab + 1

    return component_labels


def component_labels_downscaling(table: pd.DataFrame, scale_factor: float = 20) -> List[int]:
    """Label components using downscaling and connected components.

    Args:
        table: Dataframe of segmentation table.
        scale_factor: Factor for downscaling.

    Returns:
        List of component label for each point in dataframe.
    """
    labeled, largest_component = erode_sgn_seg_downscaling(table, filter_small_components=10,
                                                           scale_factor=scale_factor, threshold_erode=None)
    component_labels = get_components(table, labeled, scale_factor, distance_component=1)

    return component_labels


def postprocess_sgn_seg(
    table: pd.DataFrame,
    postprocess_type: Optional[str] = "downsampling",
) -> pd.DataFrame:
    """Postprocessing SGN segmentation of cochlea.

    Args:
        table: Dataframe of segmentation table.
        postprocess_type: Postprocessing method, either 'downsampling' or 'graph'.

    Returns:
        Dataframe with component labels.
    """
    if postprocess_type == "downsampling":
        component_labels = component_labels_downscaling(table)
    elif postprocess_type == "graph":
        component_labels = component_labels_graph(table)

    table.loc[:, "component_labels"] = component_labels

    return table
