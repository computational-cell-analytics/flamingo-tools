import numpy as np
import vigra

from skimage import measure
from scipy.spatial import distance
from scipy.sparse import csr_matrix


def filter_isolated_objects(segmentation, distance_threshold=15, neighbor_threshold=5):
    segmentation, n_ids, _ = vigra.analysis.relabelConsecutive(segmentation, start_label=1, keep_zeros=True)

    props = measure.regionprops(segmentation)
    coordinates = np.array([prop.centroid for prop in props])

    # Calculate pairwise distances and convert to a square matrix
    dist_matrix = distance.pdist(coordinates)
    dist_matrix = distance.squareform(dist_matrix)

    # Create sparse matrix of connections within the threshold distance
    sparse_matrix = csr_matrix(dist_matrix < distance_threshold, dtype=int)

    # Sum each row to count neighbors
    neighbor_counts = sparse_matrix.sum(axis=1)

    seg_ids = np.unique(segmentation)[1:]
    filter_mask = np.array(neighbor_counts < neighbor_threshold).squeeze()
    filter_ids = seg_ids[filter_mask]

    seg_filtered = segmentation.copy()
    seg_filtered[np.isin(seg_filtered, filter_ids)] = 0

    seg_filtered, n_ids_filtered, _ = vigra.analysis.relabelConsecutive(seg_filtered, start_label=1, keep_zeros=True)

    return seg_filtered, n_ids, n_ids_filtered
