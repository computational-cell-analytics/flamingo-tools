import math
import warnings
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.approximation import steiner_tree

from flamingo_tools.segmentation.postprocessing import graph_connected_components
from flamingo_tools.segmentation.distance_weighted_steiner import distance_weighted_steiner_path


def find_most_distant_nodes(G: nx.classes.graph.Graph, weight: str = 'weight') -> Tuple[float, float]:
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


def voxel_subsample(G, factor=0.25, voxel_size=None, seed=1234):
    coords = np.asarray([G.nodes[n]["pos"] for n in G.nodes])
    nodes = np.asarray(list(G.nodes))

    # choose a voxel edge length if the caller has not fixed one
    if voxel_size is None:
        bbox = np.ptp(coords, axis=0)                  # edge lengths
        voxel_size = (bbox.prod() / (len(G)/factor)) ** (1/3)

    # integer voxel indices
    mins = coords.min(axis=0)
    vox = np.floor((coords - mins) / voxel_size).astype(np.int32)

    # bucket nodes per voxel
    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, v in enumerate(map(tuple, vox)):
        buckets[v].append(idx)

    rng = np.random.default_rng(seed)
    keep = []
    for bucket in buckets.values():
        k = max(1, int(round(len(bucket)*factor)))          # local quota
        keep.extend(rng.choice(bucket, k, replace=False))

    sampled_nodes = nodes[keep]
    return G.subgraph(sampled_nodes).copy()


def measure_run_length_sgns(graph, centroids, label_ids, filter_factor, weight="weight"):
    if filter_factor is not None:
        if 0 <= filter_factor < 1:
            graph = voxel_subsample(graph, factor=filter_factor)
            centroid_labels = list(graph.nodes)
            centroids = [graph.nodes[n]["pos"] for n in graph.nodes]
            k_nn_thick = int(40 * filter_factor)
            # centroids = [centroids[label_ids.index(i)] for i in centroid_labels]

        else:
            raise ValueError(f"Invalid filter factor {filter_factor}. Choose a filter factor between 0 and 1.")
    else:
        k_nn_thick = 40
        centroid_labels = label_ids

    path_coords, path = distance_weighted_steiner_path(
            centroids,   # (N,3) ndarray
            centroid_labels=centroid_labels,  # (N,) ndarray
            k_nn_thick=k_nn_thick,      # 20‒30 is robust for SGN clouds  int(40 * (1 - filter_factor))
            lam=0.5,            # 0.3‒1.0 : larger → stronger centripetal bias
            r_connect=50.0      # connect neighbours within 50 µm
    )

    for num, p in enumerate(path[:-1]):
        pos_i = centroids[centroid_labels.index(p)]
        pos_j = centroids[centroid_labels.index(path[num+1])]
        dist = math.dist(pos_i, pos_j)
        graph.add_edge(p, path[num+1], weight=dist)

    total_distance = nx.path_weight(graph, path, weight=weight)

    return total_distance, path, graph


def measure_run_length_ihcs(graph, weight="weight"):
    u, v = find_most_distant_nodes(graph)
    # approximate Steiner tree and find shortest path between the two most distant nodes
    terminals = set(graph.nodes())  # All nodes are required
    # Approximate Steiner Tree over all nodes
    T = steiner_tree(graph, terminals, weight=weight)
    path = nx.shortest_path(T, source=u, target=v, weight=weight)
    total_distance = nx.path_weight(T, path, weight=weight)
    return total_distance, path


def map_frequency(table):
    # map frequency using Greenwood function f(x) = A * (10 **(ax) - K), for humans: a=2.1, k=0.88, A = 165.4 [kHz]
    var_k = 0.88
    # calculate values to fit (assumed) minimal (1kHz) and maximal (80kHz) hearing range of mice at x=0, x=1
    fmin = 1
    fmax = 80
    var_A = fmin / (1 - var_k)
    var_exp = ((fmax + var_A * var_k) / var_A)
    table.loc[table['distance_to_path[µm]'] >= 0, 'tonotopic_value[kHz]'] = var_A * (var_exp ** table["length_fraction"] - var_k)
    table.loc[table['distance_to_path[µm]'] < 0, 'tonotopic_value[kHz]'] = 0

    return table


def tonotopic_mapping(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    max_edge_distance: float = 30,
    min_component_length: int = 50,
    cell_type: str = "ihc",
    filter_factor: Optional[float] = None
) -> pd.DataFrame:
    """Tonotopic mapping of IHCs by supplying a table with component labels.
    The mapping assigns a tonotopic label to each IHC according to the position along the length of the cochlea.

    Args:
        table: Dataframe of segmentation table.
        component_label: List of component labels to evaluate.
        max_edge_distance: Maximal edge distance to connect nodes.
        min_component_length: Minimal number of nodes in component.
        cell_type: Cell type of segmentation.
        Filter factor: Fraction of nodes to remove before mapping.

    Returns:
        Table with tonotopic label for cells.
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    label_ids = [int(i) for i in list(new_subset["label_id"])]

    # create graph with connected components
    coords = {}
    for index, element in zip(label_ids, centroids):
        coords[index] = element

    components, graph = graph_connected_components(coords, max_edge_distance, min_component_length)
    if len(components) > 1:
        warnings.warn(f"There are {len(components)} connected components, expected 1. "
                      "Check parameters for post-processing (max_edge_distance, min_component_length).")

    unfiltered_graph = graph.copy()

    if cell_type == "ihc":
        total_distance, path = measure_run_length_ihcs(graph)

    else:
        total_distance, path, graph = measure_run_length_sgns(graph, centroids, label_ids,
                                                              filter_factor, weight="weight")

    # measure_betweenness
    centrality = nx.betweenness_centrality(graph, k=100, normalized=True, weight='weight', seed=1234)
    score = sum(centrality[n] for n in path) / len(path)
    print(f"path distance: {total_distance}")
    print(f"centrality score: {score}")

    # assign relative distance to nodes on path
    path_dict = {}
    path_dict[path[0]] = {"label_id": path[0], "length_fraction": 0}
    accumulated = 0
    for num, p in enumerate(path[1:-1]):
        distance = graph.get_edge_data(path[num], p)["weight"]
        accumulated += distance
        rel_dist = accumulated / total_distance
        path_dict[p] = {"label_id": p, "length_fraction": rel_dist}
    path_dict[path[-1]] = {"label_id": path[-1], "length_fraction": 1}

    # add missing nodes from component and compute distance to path
    pos = nx.get_node_attributes(unfiltered_graph, 'pos')
    for c in label_ids:
        if c not in path:
            min_dist = float('inf')
            nearest_node = None

            for p in path:
                dist = math.dist(pos[c], pos[p])
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = p

            path_dict[c] = {
                "label_id": c,
                "length_fraction": path_dict[nearest_node]["length_fraction"],
                "distance_to_path": min_dist,
                }
        else:
            path_dict[c]["distance_to_path"] = 0

    distance_to_path = [-1 for _ in range(len(table))]
    # 'label_id' of dataframe starting at 1
    for key in list(path_dict.keys()):
        distance_to_path[int(path_dict[key]["label_id"] - 1)] = path_dict[key]["distance_to_path"]

    table.loc[:, "distance_to_path[µm]"] = distance_to_path

    length_fraction = [0 for _ in range(len(table))]
    for key in list(path_dict.keys()):
        length_fraction[int(path_dict[key]["label_id"] - 1)] = path_dict[key]["length_fraction"]

    table.loc[:, "length_fraction"] = length_fraction
    table.loc[:, "run_length[µm]"] = table["length_fraction"] * total_distance

    table = map_frequency(table)

    return table
