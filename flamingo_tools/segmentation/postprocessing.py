import numpy as np
import vigra
import multiprocessing as mp
from concurrent import futures

from skimage import measure
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

import elf.parallel as parallel
from elf.io import open_file
import nifty.tools as nt

def distance_nearest_neighbors(tsv_table, n_neighbors=10, expand_table=True):
    """
    Calculate average distance of n nearest neighbors.

    :param DataFrame tsv_table:
    :param int n_neighbors: Number of nearest neighbors
    :param bool expand_table: Flag for expanding DataFrame
    :returns: List of average distances
    :rtype: list
    """
    centroids = list(zip(tsv_table["anchor_x"], tsv_table["anchor_y"], tsv_table["anchor_z"]))

    coordinates = np.array(centroids)

    # nearest neighbor is always itself, so n_neighbors+=1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)

    # Average distance to nearest neighbors
    distance_avg = [sum(d) / len(d) for d in distances[:, 1:]]

    if expand_table:
        tsv_table['distance_nn'+str(n_neighbors)] = distance_avg

    return distance_avg

def filter_isolated_objects(
        segmentation, output_path, tsv_table=None,
        distance_threshold=15, neighbor_threshold=5, min_size=1000,
        output_key="segmentation_postprocessed",
        ):
    """
    Postprocessing step to filter isolated objects from a segmentation.
    Instance segmentations are filtered if they have fewer neighbors than a given threshold in a given distance around them.
    Additionally, size filtering is possible if a TSV file is supplied.

    :param dataset segmentation: Dataset containing the segmentation
    :param str out_path: Output path for postprocessed segmentation
    :param str tsv_file: Optional TSV file containing segmentation parameters in MoBIE format
    :param int distance_threshold: Distance in micrometer to check for neighbors
    :param int neighbor_threshold: Minimal number of neighbors for filtering
    :param int min_size: Minimal number of pixels for filtering small instances
    :param str output_key: Output key for postprocessed segmentation
    """
    if tsv_table is not None:
        n_pixels = tsv_table["n_pixels"].to_list()
        label_ids = tsv_table["label_id"].to_list()
        centroids = list(zip(tsv_table["anchor_x"], tsv_table["anchor_y"], tsv_table["anchor_z"]))
        n_ids = len(label_ids)

        # filter out cells smaller than min_size
        if min_size is not None:
            min_size_label_ids = [l for (l,n) in zip(label_ids, n_pixels) if n <= min_size]
            centroids = [c for (c,l) in zip(centroids, label_ids) if l not in min_size_label_ids]
            label_ids = [int(l) for l in label_ids if l not in min_size_label_ids]

        coordinates = np.array(centroids)
        label_ids = np.array(label_ids)

    else:
        segmentation, n_ids, _ = vigra.analysis.relabelConsecutive(segmentation[:], start_label=1, keep_zeros=True)
        props = measure.regionprops(segmentation)
        coordinates = np.array([prop.centroid for prop in props])
        label_ids = np.unique(segmentation)[1:]

    # Calculate pairwise distances and convert to a square matrix
    dist_matrix = distance.pdist(coordinates)
    dist_matrix = distance.squareform(dist_matrix)

    # Create sparse matrix of connections within the threshold distance
    sparse_matrix = csr_matrix(dist_matrix < distance_threshold, dtype=int)

    # Sum each row to count neighbors
    neighbor_counts = sparse_matrix.sum(axis=1)

    filter_mask = np.array(neighbor_counts < neighbor_threshold).squeeze()
    filter_ids = label_ids[filter_mask]

    shape = segmentation.shape
    block_shape=(128,128,128)
    chunks=(128,128,128)

    blocking = nt.blocking([0] * len(shape), shape, block_shape)

    output = open_file(output_path, mode="a")

    output_dataset = output.create_dataset(
        output_key, shape=shape, dtype=segmentation.dtype,
        chunks=chunks, compression="gzip"
    )

    def filter_chunk(block_id):
        """
        Set all points within a chunk to zero if they match filter IDs.
        """
        block = blocking.getBlock(block_id)
        volume_index = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data = segmentation[volume_index]
        data[np.isin(data, filter_ids)] = 0
        output_dataset[volume_index] = data

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())

    with futures.ThreadPoolExecutor(n_threads) as filter_pool:
        list(tqdm(filter_pool.map(filter_chunk, range(blocking.numberOfBlocks)), total=blocking.numberOfBlocks))

    seg_filtered, n_ids_filtered, _ = parallel.relabel_consecutive(output_dataset, start_label=1, keep_zeros=True, block_shape=(128,128,128))

    return seg_filtered, n_ids, n_ids_filtered
