import math
import multiprocessing as mp
import threading
from concurrent import futures
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import elf.parallel as parallel
import numpy as np
import nifty.tools as nt
import networkx as nx
import pandas as pd

from elf.io import open_file
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from scipy.spatial import cKDTree, ConvexHull
from skimage import measure
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, watershed
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


def compute_table_on_the_fly(segmentation: np.typing.ArrayLike, resolution: float) -> pd.DataFrame:
    """Compute a segmentation table compatible with MoBIE.

    The table contains information about the number of pixels per object,
    the anchor (= centroid) and the bounding box. Anchor and bounding box are given in physical coordinates.

    Args:
        segmentation: The segmentation for which to compute the table.
        resolution: The physical voxel spacing of the data.

    Returns:
        The segmentation table.
    """
    props = measure.regionprops(segmentation)
    label_ids = np.array([prop.label for prop in props])
    coordinates = np.array([prop.centroid for prop in props]).astype("float32")
    # transform pixel distance to physical units
    coordinates = coordinates * resolution
    bb_min = np.array([prop.bbox[:3] for prop in props]).astype("float32") * resolution
    bb_max = np.array([prop.bbox[3:] for prop in props]).astype("float32") * resolution
    sizes = np.array([prop.area for prop in props])
    table = pd.DataFrame({
        "label_id": label_ids,
        "anchor_x": coordinates[:, 2],
        "anchor_y": coordinates[:, 1],
        "anchor_z": coordinates[:, 0],
        "bb_min_x": bb_min[:, 2],
        "bb_min_y": bb_min[:, 1],
        "bb_min_z": bb_min[:, 0],
        "bb_max_x": bb_max[:, 2],
        "bb_max_y": bb_max[:, 1],
        "bb_max_z": bb_max[:, 0],
        "n_pixels": sizes,
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
        The number of objects before filtering.
        The number of objects after filtering.
    """
    # Compute the table on the fly. This doesn't work for large segmentations.
    if table is None:
        table = compute_table_on_the_fly(segmentation, resolution=resolution)
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
    print(f"Initial length: {len(table)}")
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
    centroids: np.ndarray,
    scale_factor: int,
    ref_dimensions: Optional[Tuple[float, float, float]] = None,
    component_labels: Optional[List[int]] = None,
    downsample_mode: str = "accumulated",
) -> np.typing.NDArray:
    """Downscale centroids in dataframe.

    Args:
        centroids: Centroids of SGN segmentation, ndarray of shape (N, 3)
        scale_factor: Factor for downscaling coordinates.
        ref_dimensions: Reference dimensions for downscaling. Taken from centroids if not supplied.
        component_labels: List of component labels, which has to be supplied for the downsampling mode 'components'
        downsample_mode: Flag for downsampling, either 'accumulated', 'capped', or 'components'.

    Returns:
        The downscaled array
    """
    centroids_scaled = [(c[0] / scale_factor, c[1] / scale_factor, c[2] / scale_factor) for c in centroids]

    if ref_dimensions is None:
        bounding_dimensions = (max([c[0] for c in centroids]),
                               max([c[1] for c in centroids]),
                               max([c[2] for c in centroids]))
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
        if component_labels is None:
            raise KeyError("Component labels must be supplied for downsampling with mode 'components'.")
        for comp, centr in zip(component_labels, centroids_scaled):
            if comp != 0:
                new_array[int(centr[0]), int(centr[1]), int(centr[2])] = comp

    else:
        raise ValueError("Choose one of the downsampling modes 'accumulated', 'capped', or 'components'.")

    new_array = np.round(new_array).astype(int)

    return new_array


def graph_connected_components(coords: dict, max_edge_distance: float, min_component_length: int):
    """Create a list of IDs for each connected component of a graph.

    Args:
        coords: Dictionary containing label IDs as keys and their position as value.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        min_component_length: Minimal length of nodes of connected component. Filtered out if lower.

    Returns:
        List of dictionary keys of connected components.
        Graph of connected components.
    """
    graph = nx.Graph()
    for num, pos in coords.items():
        graph.add_node(num, pos=pos)

    # create edges between points whose distance is less than threshold max_edge_distance
    for num_i, pos_i in coords.items():
        for num_j, pos_j in coords.items():
            if num_i < num_j:
                dist = math.dist(pos_i, pos_j)
                if dist <= max_edge_distance:
                    graph.add_edge(num_i, num_j, weight=dist)

    components = list(nx.connected_components(graph))

    # remove connected components with less nodes than threshold min_component_length
    for component in components:
        if len(component) < min_component_length:
            for c in component:
                graph.remove_node(c)

    components = [list(s) for s in nx.connected_components(graph)]
    length_components = [len(c) for c in components]
    length_components, components = zip(*sorted(zip(length_components, components), reverse=True))

    return components, graph


def components_sgn(
    table: pd.DataFrame,
    keyword: str = "distance_nn100",
    threshold_erode: Optional[float] = None,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
    iterations_erode: Optional[int] = None,
    postprocess_threshold: Optional[float] = None,
    postprocess_components: Optional[List[int]] = None,
) -> List[List[int]]:
    """Eroding the SGN segmentation.

    Args:
        table: Dataframe of segmentation table.
        keyword: Keyword of the dataframe column for erosion.
        threshold_erode: Threshold of column value after erosion step with spatial statistics.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.
        iterations_erode: Number of iterations for erosion, normally determined automatically.
        postprocess_threshold: Post-process graph connected components by searching for points closer than threshold.
        postprocess_components: Post-process specific graph connected components ([0] for largest component only).

    Returns:
        Subgraph components as lists of label_ids of dataframe.
    """
    if keyword not in table:
        distance_avg = nearest_neighbor_distance(table, n_neighbors=100)
        table[keyword] = list(distance_avg)

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

    if iterations != 0:
        print(f"Using threshold of {threshold} micrometer for eroding segmentation with keyword {keyword}.")
        new_subset = erode_subset(table.copy(), iterations=iterations,
                                  threshold=threshold, min_cells=min_cells, keyword=keyword)
    else:
        new_subset = table.copy()

    # create graph from coordinates of eroded subset
    centroids_subset = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    labels_subset = [int(i) for i in list(new_subset["label_id"])]
    coords = {}
    for index, element in zip(labels_subset, centroids_subset):
        coords[index] = element

    components, _ = graph_connected_components(coords, max_edge_distance, min_component_length)

    # add original coordinates closer to eroded component than threshold
    if postprocess_threshold is not None:
        if postprocess_components is None:
            pp_components = components
        else:
            pp_components = [components[i] for i in postprocess_components]

        add_coords = []
        for label_id, centr in zip(labels, centroids):
            if label_id not in labels_subset:
                add_coord = []
                for comp_index, component in enumerate(pp_components):
                    for comp_label in component:
                        dist = math.dist(centr, centroids[comp_label - 1])
                        if dist <= postprocess_threshold:
                            add_coord.append([comp_index, label_id])
                            break
                if len(add_coord) != 0:
                    add_coords.append(add_coord)
        if len(add_coords) != 0:
            for c in add_coords:
                components[c[0][0]].append(c[0][1])

    return components


def label_components_sgn(
    table: pd.DataFrame,
    min_size: int = 1000,
    threshold_erode: Optional[float] = None,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
    iterations_erode: Optional[int] = None,
    postprocess_threshold: Optional[float] = None,
    postprocess_components: Optional[List[int]] = None,
) -> List[int]:
    """Label SGN components using graph connected components.

    Args:
        table: Dataframe of segmentation table.
        min_size: Minimal number of pixels for filtering small instances.
        threshold_erode: Threshold of column value after erosion step with spatial statistics.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.
        iterations_erode: Number of iterations for erosion, normally determined automatically.
        postprocess_threshold: Post-process graph connected components by searching for points closer than threshold.
        postprocess_components: Post-process specific graph connected components ([0] for largest component only).

    Returns:
        List of component label for each point in dataframe. 0 - background, then in descending order of size
    """

    # First, apply the size filter.
    entries_filtered = table[table.n_pixels < min_size]
    table = table[table.n_pixels >= min_size]

    components = components_sgn(table, threshold_erode=threshold_erode, min_component_length=min_component_length,
                                max_edge_distance=max_edge_distance, iterations_erode=iterations_erode,
                                postprocess_threshold=postprocess_threshold,
                                postprocess_components=postprocess_components)

    # add size-filtered objects to have same initial length
    table = pd.concat([table, entries_filtered], ignore_index=True)
    table.sort_values("label_id")

    component_labels = [0 for _ in range(len(table))]
    table.loc[:, "component_labels"] = component_labels
    # be aware of 'label_id' of dataframe starting at 1
    for lab, comp in enumerate(components):
        table.loc[table["label_id"].isin(comp), "component_labels"] = lab + 1

    return table


def components_ihc(
    table: pd.DataFrame,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
):
    """Create connected components for IHC segmentation.

    Args:
        table: Dataframe of segmentation table.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.

    Returns:
        Subgraph components as lists of label_ids of dataframe.
    """
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    labels = [int(i) for i in list(table["label_id"])]
    coords = {}
    for index, element in zip(labels, centroids):
        coords[index] = element

    components, _ = graph_connected_components(coords, max_edge_distance, min_component_length)
    return components


def label_components_ihc(
    table: pd.DataFrame,
    min_size: int = 1000,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
) -> List[int]:
    """Label components using graph connected components.

    Args:
        table: Dataframe of segmentation table.
        min_size: Minimal number of pixels for filtering small instances.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.

    Returns:
        List of component label for each point in dataframe. 0 - background, then in descending order of size
    """

    # First, apply the size filter.
    entries_filtered = table[table.n_pixels < min_size]
    table = table[table.n_pixels >= min_size]

    components = components_ihc(table, min_component_length=min_component_length,
                                max_edge_distance=max_edge_distance)

    # add size-filtered objects to have same initial length
    table = pd.concat([table, entries_filtered], ignore_index=True)
    table.sort_values("label_id")

    length_components = [len(c) for c in components]
    length_components, components = zip(*sorted(zip(length_components, components), reverse=True))

    component_labels = [0 for _ in range(len(table))]
    table.loc[:, "component_labels"] = component_labels
    # be aware of 'label_id' of dataframe starting at 1
    for lab, comp in enumerate(components):
        table.loc[table["label_id"].isin(comp), "component_labels"] = lab + 1

    return table


def dilate_and_trim(
    arr_orig: np.ndarray,
    edt: np.ndarray,
    iterations: int = 15,
    offset: float = 0.4,
) -> np.ndarray:
    """Dilate and trim original binary array according to a
    Euclidean Distance Trasform computed for a separate target array.

    Args:
        arr_orig: Original 3D binary array
        edt: 3D array containing Euclidean Distance transform for guiding dilation
        iterations: Number of iterations for dilations
        offset: Offset for regulating dilation. value should be in range(0, 0.45)

    Returns:
        Dilated binary array
    """
    border_coords = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(iterations):
        arr_dilated = binary_dilation(arr_orig)
        for x in range(arr_dilated.shape[0]):
            for y in range(arr_dilated.shape[1]):
                for z in range(arr_dilated.shape[2]):
                    if arr_dilated[x, y, z] != 0:
                        if arr_orig[x, y, z] == 0:
                            min_dist = float('inf')
                            for dx, dy, dz in border_coords:
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if arr_orig[nx, ny, nz] == 1:
                                    min_dist = min([min_dist, edt[nx, ny, nz]])
                            if edt[x, y, z] >= min_dist - offset:
                                arr_dilated[x, y, z] = 0
        arr_orig = arr_dilated
    return arr_dilated


def filter_cochlea_volume_single(
    table: pd.DataFrame,
    components: Optional[List[int]] = [1],
    scale_factor: int = 48,
    resolution: float = 0.38,
    dilation_iterations: int = 12,
    padding: int = 1200,
) -> np.ndarray:
    """Filter cochlea volume based on a segmentation table.
    Centroids contained in the segmentation table are used to create a down-scaled binary array.
    The array can be dilated.

    Args:
        table: Segmentation table.
        components: Component labels for filtering segmentation table.
        scale_factor: Down-sampling factor for filtering.
        resolution: Resolution of pixel in µm.
        dilation_iterations: Iterations for dilating binary segmentation mask. A negative value omits binary closing.
        padding: Padding in pixel to apply to guessed dimensions based on centroid coordinates.

    Returns:
        Binary 3D array of filtered cochlea.
    """
    # filter components
    if components is not None:
        table = table[table["component_labels"].isin(components)]

    # identify approximate input dimensions for down-scaling
    centroids = list(zip(table["anchor_x"] / resolution,
                         table["anchor_y"] / resolution,
                         table["anchor_z"] / resolution))

    # padding the array allows for dilation without worrying about array borders
    max_x = table["anchor_x"].max() / resolution + padding
    max_y = table["anchor_y"].max() / resolution + padding
    max_z = table["anchor_z"].max() / resolution + padding
    ref_dimensions = (max_x, max_y, max_z)

    # down-scale arrays
    array_downscaled = downscaled_centroids(centroids, ref_dimensions=ref_dimensions,
                                            scale_factor=scale_factor, downsample_mode="capped")

    if dilation_iterations > 0:
        array_dilated = binary_dilation(array_downscaled, np.ones((3, 3, 3)), iterations=dilation_iterations)
        return binary_closing(array_dilated, np.ones((3, 3, 3)), iterations=1)

    elif dilation_iterations == 0:
        return binary_closing(array_downscaled, np.ones((3, 3, 3)), iterations=1)

    else:
        return array_downscaled


def filter_cochlea_volume(
    sgn_table: pd.DataFrame,
    ihc_table: pd.DataFrame,
    sgn_components: Optional[List[int]] = [1],
    ihc_components: Optional[List[int]] = [1],
    scale_factor: int = 48,
    resolution: float = 0.38,
    dilation_iterations: int = 12,
    padding: int = 1200,
    dilation_method: str = "individual",
) -> np.ndarray:
    """Filter cochlea volume with SGN and IHC segmentation.
    Centroids contained in the segmentation tables are used to create down-scaled binary arrays.
    The arrays are then dilated using guided dilation to fill the section inbetween SGNs and IHCs.

    Args:
        sgn_table: SGN segmentation table.
        ihc_table: IHC segmentation table.
        sgn_components: Component labels for filtering SGN segmentation table.
        ihc_components: Component labels for filtering IHC segmentation table.
        scale_factor: Down-sampling factor for filtering.
        resolution: Resolution of pixel in µm.
        dilation_iterations: Iterations for dilating binary segmentation mask.
        padding: Padding in pixel to apply to guessed dimensions based on centroid coordinates.
        dilation_method: Dilation style for SGN and IHC segmentation, either 'individual', 'combined' or no dilation.

    Returns:
        Binary 3D array of filtered cochlea.
    """
    # filter components
    if sgn_components is not None:
        sgn_table = sgn_table[sgn_table["component_labels"].isin(sgn_components)]
    if ihc_components is not None:
        ihc_table = ihc_table[ihc_table["component_labels"].isin(ihc_components)]

    # identify approximate input dimensions for down-scaling
    centroids_sgn = list(zip(sgn_table["anchor_x"] / resolution,
                             sgn_table["anchor_y"] / resolution,
                             sgn_table["anchor_z"] / resolution))
    centroids_ihc = list(zip(ihc_table["anchor_x"] / resolution,
                             ihc_table["anchor_y"] / resolution,
                             ihc_table["anchor_z"] / resolution))

    # padding the array allows for dilation without worrying about array borders
    max_x = max([sgn_table["anchor_x"].max(), ihc_table["anchor_x"].max()]) / resolution + padding
    max_y = max([sgn_table["anchor_y"].max(), ihc_table["anchor_y"].max()]) / resolution + padding
    max_z = max([sgn_table["anchor_z"].max(), ihc_table["anchor_z"].max()]) / resolution + padding
    ref_dimensions = (max_x, max_y, max_z)

    # down-scale arrays
    array_downscaled_sgn = downscaled_centroids(centroids_sgn, ref_dimensions=ref_dimensions,
                                                scale_factor=scale_factor, downsample_mode="capped")

    array_downscaled_ihc = downscaled_centroids(centroids_ihc, ref_dimensions=ref_dimensions,
                                                scale_factor=scale_factor, downsample_mode="capped")

    # dilate down-scaled SGN array in direction of IHC segmentation
    distance_from_sgn = distance_transform_edt(~array_downscaled_sgn.astype(bool))
    iterations = 20
    arr_dilated = dilate_and_trim(array_downscaled_ihc.copy(), distance_from_sgn, iterations=iterations, offset=0.4)

    # dilate single structures first
    if dilation_method == "individual":
        ihc_dilated = binary_dilation(array_downscaled_ihc, np.ones((3, 3, 3)), iterations=dilation_iterations)
        sgn_dilated = binary_dilation(array_downscaled_sgn, np.ones((3, 3, 3)), iterations=dilation_iterations)
        combined_dilated = arr_dilated + ihc_dilated + sgn_dilated
        combined_dilated[combined_dilated > 0] = 1
        combined_dilated = binary_dilation(combined_dilated, np.ones((3, 3, 3)), iterations=1)

    # dilate combined structure
    elif dilation_method == "combined":
        # combine SGN, IHC, and region between both to form output mask
        combined_structure = arr_dilated + array_downscaled_ihc + array_downscaled_sgn
        combined_structure[combined_structure > 0] = 1
        combined_dilated = binary_dilation(combined_structure, np.ones((3, 3, 3)), iterations=dilation_iterations)

    # no dilation of combined structure
    else:
        combined_dilated = arr_dilated + ihc_dilated + sgn_dilated
        combined_dilated[combined_dilated > 0] = 1

    return combined_dilated


def split_nonconvex_objects(
    segmentation: np.typing.ArrayLike,
    output: np.typing.ArrayLike,
    segmentation_table: pd.DataFrame,
    min_size: int,
    resolution: Union[float, Sequence[float]],
    height_map: Optional[np.typing.ArrayLike] = None,
    component_labels: Optional[List[int]] = None,
    n_threads: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Split noncovex objects into multiple parts inplace.

    Args:
        segmentation:
        output:
        segmentation_table:
        min_size:
        resolution:
        height_map:
        component_labels:
        n_threads:
    """
    if isinstance(resolution, float):
        resolution = [resolution] * 3
    assert len(resolution) == 3
    resolution = np.array(resolution)

    lock = threading.Lock()
    offset = len(segmentation_table)

    def split_object(object_id):
        nonlocal offset

        row = segmentation_table[segmentation_table.label_id == object_id]
        if row.n_pixels.values[0] < min_size:
            # print(object_id, ": min-size")
            return [object_id]

        bb_min = np.array([
            row.bb_min_z.values[0], row.bb_min_y.values[0], row.bb_min_x.values[0],
        ]) / resolution
        bb_max = np.array([
            row.bb_max_z.values[0], row.bb_max_y.values[0], row.bb_max_x.values[0],
        ]) / resolution

        bb_min = np.maximum(bb_min.astype(int) - 1, np.array([0, 0, 0]))
        bb_max = np.minimum(bb_max.astype(int) + 1, np.array(list(segmentation.shape)))
        bb = tuple(slice(mi, ma) for mi, ma in zip(bb_min, bb_max))

        # This is due to segmentation artifacts.
        bb_shape = bb_max - bb_min
        if (bb_shape > 500).any():
            print(object_id, "has a too large shape:", bb_shape)
            return [object_id]

        seg = segmentation[bb]
        mask = ~find_boundaries(seg)
        dist = distance_transform_edt(mask, sampling=resolution)

        seg_mask = seg == object_id
        dist[~seg_mask] = 0
        dist = gaussian(dist, (0.6, 1.2, 1.2))
        maxima = peak_local_max(dist, min_distance=3, exclude_border=True)

        if len(maxima) == 1:
            # print(object_id, ": max len")
            return [object_id]

        with lock:
            old_offset = offset
            offset += len(maxima)

        seeds = np.zeros(seg.shape, dtype=int)
        for i, pos in enumerate(maxima, 1):
            seeds[tuple(pos)] = old_offset + i

        if height_map is None:
            hmap = dist.max() - dist
        else:
            hmap = height_map[bb]
        new_seg = watershed(hmap, markers=seeds, mask=seg_mask)

        seg_ids, sizes = np.unique(new_seg, return_counts=True)
        seg_ids, sizes = seg_ids[1:], sizes[1:]

        keep_ids = seg_ids[sizes > min_size]
        if len(keep_ids) < 2:
            # print(object_id, ": keep-id")
            return [object_id]

        elif len(keep_ids) != len(seg_ids):
            new_seg[~np.isin(new_seg, keep_ids)] = 0
            new_seg = watershed(hmap, markers=new_seg, mask=seg_mask)

        with lock:
            out = output[bb]
            out[seg_mask] = new_seg[seg_mask]
            output[bb] = out

        # print(object_id, ":", len(keep_ids))
        return keep_ids.tolist()

        # import napari
        # v = napari.Viewer()
        # v.add_image(hmap)
        # v.add_labels(seg)
        # v.add_labels(new_seg)
        # v.add_points(maxima)
        # napari.run()

    if component_labels is None:
        object_ids = segmentation_table.label_id.values
    else:
        object_ids = segmentation_table[segmentation_table.component_labels.isin(component_labels)].label_id.values

    if n_threads is None:
        n_threads = mp.cpu_count()

    # new_id_mapping = []
    # for object_id in tqdm(object_ids, desc="Split non-convex objects"):
    #     new_id_mapping.append(split_object(object_id))

    with futures.ThreadPoolExecutor(n_threads) as tp:
        new_id_mapping = list(
            tqdm(tp.map(split_object, object_ids), total=len(object_ids), desc="Split non-convex objects")
        )

    new_id_mapping = {object_id: mapped_ids for object_id, mapped_ids in zip(object_ids, new_id_mapping)}
    return new_id_mapping
