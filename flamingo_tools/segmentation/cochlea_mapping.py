import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from flamingo_tools.segmentation.postprocessing import graph_connected_components


def find_most_distant_nodes(G, weight='weight'):
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


def steiner_path_between_distant_nodes(G, weight='weight'):
    # Step 1: Find the most distant pair of nodes
    u, v = find_most_distant_nodes(G, weight=weight)
    terminals = set(G.nodes())  # All nodes are required

    # Step 2: Approximate Steiner Tree over all nodes
    T = steiner_tree(G, terminals, weight=weight)

    # Step 3: Find the shortest path between u and v in the Steiner Tree
    path = nx.shortest_path(T, source=u, target=v, weight=weight)
    total_weight = nx.path_weight(T, path, weight=weight)

    return {
        "start": u,
        "end": v,
        "path": path,
        "total_weight": total_weight,
        "steiner_tree": T
    }


def nearest_node_on_path(G, main_path, query_node, weight='weight'):
    """Find the nearest node in the connected component graph,
    which lies on the path between the two most distant nodes.
    """
    if query_node in main_path:
        return {
            "nearest_node": query_node,
            "distance": 0
        }

    min_dist = float('inf')
    nearest_node = None

    for path_node in main_path:
        try:
            dist = nx.dijkstra_path_length(G, source=query_node, target=path_node, weight=weight)
            if dist < min_dist:
                min_dist = dist
                nearest_node = path_node
        except nx.NetworkXNoPath:
            continue  # No path to this node

    return {
        "nearest_node": nearest_node,
        "distance": min_dist if nearest_node is not None else None
    }


def tonotopic_mapping(table, component_label=[1], min_edge_distance=30, min_component_length=50,
                      cell_type="ihc"):
    """Tonotopic mapping of IHCs by supplying a table with component labels.
    The mapping assigns a tonotopic label to each IHC according to the position along the length of the cochlea.
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    comp_label_ids = list(new_subset["label_id"])
    centroids_subset = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    labels_subset = [int(i) for i in list(new_subset["label_id"])]

    # create graph with connected components
    coords = {}
    for index, element in zip(labels_subset, centroids_subset):
        coords[index] = element

    components, graph = graph_connected_components(coords, min_edge_distance, min_component_length)

    # approximate Steiner tree and find shortest path between the two most distant nodes

    u, v = find_most_distant_nodes(graph)
    if cell_type == "ihc":
        terminals = set(graph.nodes())  # All nodes are required
        # Approximate Steiner Tree over all nodes
        T = steiner_tree(graph, terminals)
        path = nx.shortest_path(T, source=u, target=v)
        total_distance = nx.path_weight(T, path)

    else:
        path = nx.shortest_path(graph, source=u, target=v)
        total_distance = nx.path_weight(graph, path)

    # assign relative distance to nodes on path
    path_list = []
    path_list.append({"label_id": path[0], "value": 0})
    accumulated = 0
    for num, p in enumerate(path[1:-1]):
        distance = graph.get_edge_data(path[num], p)["weight"]
        accumulated += distance
        rel_dist = accumulated / total_distance
        path_list.append({"label_id": p, "value": rel_dist})
    path_list.append({"label_id": path[-1], "value": 1})

    # add missing nodes from component
    for c in comp_label_ids:
        if c not in path:
            nearest_node = nearest_node_on_path(graph, path, c)["nearest_node"]
            for label in path_list:
                if label["label_id"] == nearest_node:
                    nearest_node_value = label["value"]
                    continue
            path_list.append({"label_id": int(c), "value": nearest_node_value})

    tonotopic = [0 for _ in range(len(table))]
    # be aware of 'label_id' of dataframe starting at 1
    for d in path_list:
        tonotopic[d["label_id"] - 1] = d["value"] * len(total_distance)

    table.loc[:, "tonotopic_label"] = tonotopic

    return table
