import math
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing
from scipy.interpolate import interp1d

from flamingo_tools.segmentation.postprocessing import downscaled_centroids


def find_most_distant_nodes(G: nx.classes.graph.Graph, weight: str = 'weight') -> Tuple[float, float]:
    """Find the most distant nodes in a graph.

    Args:
        G: Input graph.

    Returns:
        Node 1.
        Node 2.
    """
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    max_dist = 0
    farthest_pair = (None, None)

    for u, dist_dict in all_lengths.items():
        for v, d in dist_dict.items():
            if d > max_dist:
                max_dist = d
                farthest_pair = (u, v)

    u, v = farthest_pair
    return u, v


def central_path_edt_graph(mask: np.ndarray, start: Tuple[int], end: Tuple[int]):
    """Find the central path within a binary mask between a start and an end coordinate.

    Args:
        mask: Binary mask of volume.
        start: Starting coordinate.
        end: End coordinate.

    Returns:
        Coordinates of central path.
    """
    dt = distance_transform_edt(mask)
    G = nx.Graph()
    shape = mask.shape
    def idx_to_node(z, y, x): return z*shape[1]*shape[2] + y*shape[2] + x
    border_coords = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if not mask[z, y, x]:
                    continue
                u = idx_to_node(z, y, x)
                for dz, dy, dx in border_coords:
                    nz, ny, nx_ = z+dz, y+dy, x+dx
                    if nz >= 0 and nz < shape[0] and mask[nz, ny, nx_]:
                        v = idx_to_node(nz, ny, nx_)
                        w = 1.0 / (1e-3 + min(dt[z, y, x], dt[nz, ny, nx_]))
                        G.add_edge(u, v, weight=w)
    s = idx_to_node(*start)
    t = idx_to_node(*end)
    path = nx.shortest_path(G, source=s, target=t, weight="weight")
    coords = [(p//(shape[1]*shape[2]),
               (p//shape[2]) % shape[1],
               p % shape[2]) for p in path]
    return np.array(coords)


def moving_average_3d(path: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 3D path with a simple moving average filter.

    Args:
        path: ndarray of shape (N, 3).
        window: half-window size; actual window = 2*window + 1.

    Returns:
        smoothed path: ndarray of same shape.
    """
    kernel_size = 2 * window + 1
    kernel = np.ones(kernel_size) / kernel_size

    smooth_path = np.zeros_like(path)

    for d in range(3):
        pad = np.pad(path[:, d], window, mode='edge')
        smooth_path[:, d] = np.convolve(pad, kernel, mode='valid')

    return smooth_path


def measure_run_length_sgns(
        centroids: np.ndarray,
        scale_factor: int = 10,
        apex_higher: bool = True,
) -> Tuple[float, np.ndarray, dict]:
    """Measure the run lengths of the SGN segmentation by finding a central path through Rosenthal's canal.
    1) Create a binary mask based on down-scaled centroids.
    2) Dilate the mask and close holes to ensure a filled structure.
    3) Determine the endpoints of the structure using the principal axis.
    4) Identify a central path based on the 3D Euclidean distance transform.
    5) The path is up-scaled and smoothed using a moving average filter.
    6) The points of the path are fed into a dictionary along with the fractional length.

    Args:
        centroids: Centroids of the SGN segmentation, ndarray of shape (N, 3).
        scale_factor: Downscaling factor for finding the central path.
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.

    Returns:
        Total distance of the path.
        Path as an nd.array of positions.
        A dictionary containing the position and the length fraction of each point in the path.
    """
    mask = downscaled_centroids(centroids, scale_factor=scale_factor, downsample_mode="capped")
    mask = binary_dilation(mask, np.ones((3, 3, 3)), iterations=1)
    mask = binary_closing(mask, np.ones((3, 3, 3)), iterations=1)
    pts = np.argwhere(mask == 1)

    # find two endpoints: min/max along principal axis
    c_mean = pts.mean(axis=0)
    cov = np.cov((pts-c_mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    proj = (pts - c_mean) @ axis
    start_voxel = tuple(pts[proj.argmin()])
    end_voxel = tuple(pts[proj.argmax()])

    # compare y-value to not get into confusion with MoBIE dimensions
    if start_voxel[1] > end_voxel[1]:
        apex = start_voxel if apex_higher else end_voxel
        base = end_voxel if apex_higher else start_voxel
    else:
        apex = end_voxel if apex_higher else start_voxel
        base = start_voxel if apex_higher else end_voxel

    # get central path and total distance
    path = central_path_edt_graph(mask, apex, base)
    path = path * scale_factor
    path = moving_average_3d(path, window=5)
    total_distance = sum([math.dist(path[num + 1], path[num]) for num in range(len(path) - 1)])

    # assign relative distance to points on path
    path_dict = {}
    path_dict[0] = {"pos": path[0], "length_fraction": 0}
    accumulated = 0
    for num, p in enumerate(path[1:-1]):
        distance = math.dist(path[num], p)
        accumulated += distance
        rel_dist = accumulated / total_distance
        path_dict[num + 1] = {"pos": p, "length_fraction": rel_dist}
    path_dict[len(path)] = {"pos": path[-1], "length_fraction": 1}

    return total_distance, path, path_dict


def measure_run_length_ihcs(
    centroids: np.ndarray,
    max_edge_distance: float = 50,
    apex_higher: bool = True,
) -> Tuple[float, np.ndarray, dict]:
    """Measure the run lengths of the IHC segmentation
    by determining the shortest path between the most distant nodes of a graph.
    The graph is created based on a maximal edge distance between nodes.
    However, this value may not correspond to the max_edge_distance parameter used to process the IHC segmentation,
    which may consist of more than one component.
    That's why a large max_edge_distance is used to connect different components and identify
    the two most distant nodes and the shortest path between them.
    The path is then edited using a moving average filter.

    Args:
        centroids: Centroids of SGN segmentation.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.

    Returns:
        Total distance of the path.
        Path as an nd.array of positions.
        A dictionary containing the position and the length fraction of each point in the path.
    """
    graph = nx.Graph()
    coords = {}
    labels = [int(i) for i in range(len(centroids))]
    for index, element in zip(labels, centroids):
        coords[index] = element

    for num, pos in coords.items():
        graph.add_node(num, pos=pos)

    # TODO: Find cleaner option than the creation of a new graph with edges
    # to identify start / end point and shortest path.

    # create edges between points whose distance is less than threshold max_edge_distance
    for num_i, pos_i in coords.items():
        for num_j, pos_j in coords.items():
            if num_i < num_j:
                dist = math.dist(pos_i, pos_j)
                if dist <= max_edge_distance:
                    graph.add_edge(num_i, num_j, weight=dist)

    start_node, end_node = find_most_distant_nodes(graph)

    # compare y-value to not get into confusion with MoBIE dimensions
    if graph.nodes[start_node]["pos"][1] > graph.nodes[end_node]["pos"][1]:
        apex_node = start_node if apex_higher else end_node
        base_node = end_node if apex_higher else start_node
    else:
        apex_node = end_node if apex_higher else start_node
        base_node = start_node if apex_higher else end_node

    path = nx.shortest_path(graph, source=apex_node, target=base_node)
    total_distance = nx.path_weight(graph, path, weight="weight")

    # assign relative distance to points on path
    path_dict = {}
    path_dict[0] = {"pos": graph.nodes[path[0]]["pos"], "length_fraction": 0}
    accumulated = 0
    for num, p in enumerate(path[1:-1]):
        distance = math.dist(graph.nodes[path[num]]["pos"], graph.nodes[p]["pos"])
        accumulated += distance
        rel_dist = accumulated / total_distance
        path_dict[num + 1] = {"pos": graph.nodes[p]["pos"], "length_fraction": rel_dist}
    path_dict[len(path)] = {"pos": graph.nodes[path[-1]]["pos"], "length_fraction": 1}

    path_pos = np.array([graph.nodes[p]["pos"] for p in path])
    path = moving_average_3d(path_pos, window=5)

    return total_distance, path, path_dict


def map_frequency(table: pd.DataFrame, cell_type: str, animal: str = "mouse"):
    """Map the frequency range of SGNs in the cochlea
    using Greenwood function f(x) = A * (10 **(ax) - K).
    Values for humans: a=2.1, k=0.88, A = 165.4 [kHz].
    For mice: fit values between minimal (1kHz) and maximal (80kHz) values

    Args:
        table: Dataframe containing the segmentation.

    Returns:
        Dataframe containing frequency in an additional column 'frequency[kHz]'.
    """
    if animal == "mouse":
        if cell_type == "ihc":
            # freq_min = 5.16 kHz
            # freq_max = 81.38 kHz
            var_A = 4.232
            var_a = 1.279
            var_k = -0.22
        if cell_type == "sgn":
            # freq_min = 0.0095 kHz
            # freq_max = 47.47 kHz
            var_A = 0.38
            var_a = 2.1
            var_k = 0.975

    elif animal == "gerbil":
        if cell_type == "ihc":
            # freq_min = 0.0105 kHz
            # freq_max = 43.82 kHz
            var_A = 0.35
            var_a = 2.1
            var_k = 0.7
        if cell_type == "sgn":
            # freq_min = 0.0105 kHz
            # freq_max = 43.82 kHz
            var_A = 0.35
            var_a = 2.1
            var_k = 0.7

        # alternative Gerbil Greenwood function according to Mueller1995
        # var_A = 0.398
        # var_a = 2.2
        # var_k = 0.631
    else:
        raise ValueError("Animal not supported. Use either 'mouse' or 'gerbil'.")

    table.loc[table['offset'] >= 0, 'frequency[kHz]'] = var_A * (10 ** (var_a * table["length_fraction"]) - var_k)
    table.loc[table['offset'] < 0, 'frequency[kHz]'] = 0

    return table


def equidistant_centers(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    cell_type: str = "sgn",
    n_blocks: int = 10,
    offset_blocks: bool = True,
) -> np.ndarray:
    """Find equidistant centers within the central path of the Rosenthal's canal.

    Args:
        table: Dataframe containing centroids of SGN segmentation.
        component_label: List of components for centroid subset.
        cell_type: Cell type of the segmentation.
        n_blocks: Number of equidistant centers for block creation.
        offset_block: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.

    Returns:
        Equidistant centers as float values
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))

    if cell_type == "ihc":
        total_distance, path, _ = measure_run_length_ihcs(centroids)

    else:
        total_distance, path, _ = measure_run_length_sgns(centroids)

    diffs = np.diff(path, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.insert(np.cumsum(seg_lens), 0, 0)
    if offset_blocks:
        target_s = np.linspace(0, total_distance, n_blocks * 2 + 1)
        target_s = [s for num, s in enumerate(target_s) if num % 2 == 1]
    else:
        target_s = np.linspace(0, total_distance, n_blocks)
    f = interp1d(cum_len, path, axis=0)
    centers = f(target_s)
    return centers


def tonotopic_mapping(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    cell_type: str = "ihc",
    animal: str = "mouse",
) -> pd.DataFrame:
    """Tonotopic mapping of IHCs by supplying a table with component labels.
    The mapping assigns a tonotopic label to each IHC according to the position along the length of the cochlea.

    Args:
        table: Dataframe of segmentation table.
        component_label: List of component labels to evaluate.
        cell_type: Cell type of segmentation.

    Returns:
        Table with tonotopic label for cells.
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]

    # option for filtering IHC instances without synapses
    # leaving it commented for now because it would have little effect

    # if "syn_per_IHC" in new_subset.columns:
    #    syn_limit = 0
    #    print(f"Keeping IHC instances with more than {syn_limit} synapses.")
    #    new_subset = new_subset[new_subset["syn_per_IHC"] > syn_limit]

    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    label_ids = [int(i) for i in list(new_subset["label_id"])]

    if cell_type == "ihc":
        total_distance, _, path_dict = measure_run_length_ihcs(centroids)

    else:
        total_distance, _, path_dict = measure_run_length_sgns(centroids)

    # add missing nodes from component and compute distance to path
    node_dict = {}
    for num, c in enumerate(label_ids):
        min_dist = float('inf')
        nearest_node = None

        for key in path_dict.keys():
            dist = math.dist(centroids[num], path_dict[key]["pos"])
            if dist < min_dist:
                min_dist = dist
                nearest_node = key

        node_dict[c] = {
            "label_id": c,
            "length_fraction": path_dict[nearest_node]["length_fraction"],
            "offset": min_dist,
            }

    offset = [-1 for _ in range(len(table))]
    offset = list(np.float64(offset))
    table.loc[:, "offset"] = offset
    # 'label_id' of dataframe starting at 1
    for key in list(node_dict.keys()):
        table.loc[table["label_id"] == key, "offset"] = node_dict[key]["offset"]

    length_fraction = [0 for _ in range(len(table))]
    length_fraction = list(np.float64(length_fraction))
    table.loc[:, "length_fraction"] = length_fraction
    for num, key in enumerate(list(node_dict.keys())):
        table.loc[table["label_id"] == key, "length_fraction"] = node_dict[key]["length_fraction"]

    table.loc[:, "length[µm]"] = table["length_fraction"] * total_distance

    table = map_frequency(table, cell_type=cell_type, animal=animal)

    return table
