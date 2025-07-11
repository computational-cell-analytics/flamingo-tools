import math
import warnings
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.approximation import steiner_tree

from flamingo_tools.segmentation.postprocessing import graph_connected_components


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
    weight = "weight"
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    comp_label_ids = list(new_subset["label_id"])
    centroids_subset = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    labels_subset = [int(i) for i in list(new_subset["label_id"])]

    # create graph with connected components
    coords = {}
    for index, element in zip(labels_subset, centroids_subset):
        coords[index] = element

    components, graph = graph_connected_components(coords, max_edge_distance, min_component_length)
    if len(components) > 1:
        warnings.warn(f"There are {len(components)} connected components, expected 1. "
                      "Check parameters for post-processing (max_edge_distance, min_component_length).")

    unfiltered_graph = graph.copy()

    if filter_factor is not None:
        if 0 <= filter_factor < 1:
            rng = np.random.default_rng(seed=1234)
            original_array = np.array(comp_label_ids)
            target_length = int(len(original_array) * filter_factor)
            filtered_list = list(rng.choice(original_array, size=target_length, replace=False))
            for filter_id in filtered_list:
                graph.remove_node(filter_id)
        else:
            raise ValueError(f"Invalid filter factor {filter_factor}. Choose a filter factor between 0 and 1.")

    u, v = find_most_distant_nodes(graph)

    if not nx.has_path(graph, source=u, target=v) or cell_type == "ihc":
        # approximate Steiner tree and find shortest path between the two most distant nodes
        terminals = set(graph.nodes())  # All nodes are required
        # Approximate Steiner Tree over all nodes
        T = steiner_tree(graph, terminals, weight=weight)
        path = nx.shortest_path(T, source=u, target=v, weight=weight)
        total_distance = nx.path_weight(T, path, weight=weight)

    else:
        path = nx.shortest_path(graph, source=u, target=v, weight=weight)
        total_distance = nx.path_weight(graph, path, weight=weight)

    # assign relative distance to nodes on path
    path_list = {}
    path_list[path[0]] = {"label_id": path[0], "tonotopic": 0}
    accumulated = 0
    for num, p in enumerate(path[1:-1]):
        distance = graph.get_edge_data(path[num], p)["weight"]
        accumulated += distance
        rel_dist = accumulated / total_distance
        path_list[p] = {"label_id": p, "tonotopic": rel_dist}
    path_list[path[-1]] = {"label_id": path[-1], "tonotopic": 1}

    # add missing nodes from component
    pos = nx.get_node_attributes(unfiltered_graph, 'pos')
    for c in comp_label_ids:
        if c not in path:
            min_dist = float('inf')
            nearest_node = None

            for p in path:
                dist = math.dist(pos[c], pos[p])
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = p

            path_list[c] = {"label_id": c, "tonotopic": path_list[nearest_node]["tonotopic"]}

    # label in micrometer
    tonotopic = [0 for _ in range(len(table))]
    # be aware of 'label_id' of dataframe starting at 1
    for key in list(path_list.keys()):
        tonotopic[int(path_list[key]["label_id"] - 1)] = path_list[key]["tonotopic"] * total_distance

    table.loc[:, "tonotopic_label"] = tonotopic

    # map frequency using Greenwood function f(x) = A * (10 **(ax) - K), for humans: a=2.1, k=0.88, A = 165.4 [kHz]
    tonotopic_map = [0 for _ in range(len(table))]
    var_k = 0.88
    # calculate values to fit (assumed) minimal (1kHz) and maximal (80kHz) hearing range of mice at x=0, x=1
    fmin = 1
    fmax = 80
    var_A = fmin / (1 - var_k)
    var_exp = ((fmax + var_A * var_k) / var_A)
    for key in list(path_list.keys()):
        tonotopic_map[int(path_list[key]["label_id"] - 1)] = var_A * (var_exp ** path_list[key]["tonotopic"] - var_k)

    table.loc[:, "tonotopic_value[kHz]"] = tonotopic_map

    return table
