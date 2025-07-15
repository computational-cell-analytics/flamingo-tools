"""
distance_weighted_steiner.py
Variant-B: centre-seeking Steiner path for cochlear run-length extraction
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from typing import Tuple, Sequence, Optional


def estimate_local_thickness(points: np.ndarray,
                             k_nn: int = 20) -> np.ndarray:
    """
    Return a per-point scalar proportional to local canal thickness.
    We use the *k*-th NN distance as a cheap proxy.
    """
    tree = cKDTree(points)
    # distances shape → (N, k_nn)
    dists, _ = tree.query(points, k=k_nn + 1)   # +1 because k=0 is the point itself
    kth = dists[:, -1]                          # farthest of the k neighbours
    return kth                                  # units: same as points


def make_graph(points: np.ndarray,
               radii: np.ndarray,
               r_connect: float = 60.0,
               lam: float = 0.5,
               k_edge: Optional[int] = None) -> nx.Graph:
    """
    Build a graph with distance-transform-weighted edges.

    Parameters
    ----------
    points   : (N,3) float array
    radii    : (N,) local thickness proxy
    r_connect: connect all neighbours within this radius (µm)
    lam      : weight of |d_i - d_j| term
    k_edge   : alternative to r_connect - connect the k_edge
               nearest neighbours; leave None to use radius
    """
    N = len(points)
    tree = cKDTree(points)

    G = nx.Graph()
    # add nodes with attributes
    for idx, (xyz, r) in enumerate(zip(points, radii)):
        G.add_node(idx, pos=tuple(xyz), radius=float(r))

    # choose connectivity strategy
    if k_edge is not None:
        for idx in range(N):
            _, inds = tree.query(points[idx], k=k_edge + 1)
            for j in inds[1:]:
                _add_edge(G, idx, j, radii, lam)
    else:
        # radius search in batches (memory safe)
        pairs = tree.query_pairs(r_connect)
        for i, j in pairs:
            _add_edge(G, i, j, radii, lam)

    return G


def _add_edge(G: nx.Graph, i: int, j: int,
              radii: np.ndarray, lam: float):
    """Helper to compute weighted edge once and add both directions."""
    pi, pj = G.nodes[i]["pos"], G.nodes[j]["pos"]
    dij = np.linalg.norm(np.subtract(pi, pj))
    dr = abs(radii[i] - radii[j]) / (radii[i] + radii[j] + 1e-9)
    w = dij * (1.0 + lam * dr)
    G.add_edge(i, j, weight=w)


def find_endpoints(points: np.ndarray) -> Tuple[int, int]:
    """
    Pick apical+basal terminals as the points with minimum/maximum
    projection on the first PCA axis (fast & robust).
    """
    # simple PCA via SVD
    pts = points - points.mean(0, keepdims=True)
    u, s, vh = np.linalg.svd(pts, full_matrices=False)
    axis = vh[0]
    proj = pts @ axis
    return int(proj.argmin()), int(proj.argmax())


def distance_weighted_steiner_path(centroids: Sequence[Sequence[float]],
                                   *,
                                   centroid_labels: Optional[Sequence[int]] = None,
                                   k_nn_thick: int = 20,
                                   lam: float = 0.5,
                                   r_connect: float = 60.0,
                                   k_edge: Optional[int] = None) -> Tuple[np.ndarray, list[int]]:
    """
    Main public entry - returns (Mx3 point array, list of node indices)
    representing the centre-biased cochlear path.
    """
    pts = np.asarray(centroids, dtype=float)
    radii = estimate_local_thickness(pts, k_nn=k_nn_thick)

    G = make_graph(pts, radii, r_connect=r_connect, lam=lam, k_edge=k_edge)

    s, t = find_endpoints(pts)
    steiner = nx.algorithms.approximation.steinertree.steiner_tree(G, {s, t}, weight="weight")
    # unique s–t path inside the tree (no branches because only 2 terminals):
    path_nodes = nx.shortest_path(steiner, source=s, target=t, weight="weight")
    path_xyz = np.array([G.nodes[i]["pos"] for i in path_nodes])

    # transfer path nodes into centroid_labels
    if centroid_labels is not None:
        path_nodes = [centroid_labels[i] for i in path_nodes]

    return path_xyz, path_nodes
