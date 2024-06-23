import argparse
import os

import numpy as np
import vigra
import z5py
from elf.io import open_file

from skimage import measure
from scipy.spatial import distance
from scipy.sparse import csr_matrix


def filter_segmentation(seg_path, seg_key):
    print("Loading segmentation ...")
    with open_file(seg_path, "r") as f:
        seg = f[seg_key][:]

    seg, n_ids, _ = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)
    print("Number of nuclei:", n_ids)

    props = measure.regionprops(seg)
    coordinates = np.array([prop.centroid for prop in props])

    # Threshold distance
    threshold_distance = 15
    neighbor_threshold = 5

    # Calculate pairwise distances and convert to a square matrix
    dist_matrix = distance.pdist(coordinates)
    dist_matrix = distance.squareform(dist_matrix)

    # Create sparse matrix of connections within the threshold distance
    sparse_matrix = csr_matrix(dist_matrix < threshold_distance, dtype=int)

    # Sum each row to count neighbors
    neighbor_counts = sparse_matrix.sum(axis=1)

    seg_ids = np.unique(seg)[1:]
    filter_mask = np.array(neighbor_counts < neighbor_threshold).squeeze()
    filter_ids = seg_ids[filter_mask]

    seg_filtered = seg.copy()
    seg_filtered[np.isin(seg_filtered, filter_ids)] = 0

    seg_filtered, n_ids, _ = vigra.analysis.relabelConsecutive(seg_filtered, start_label=1, keep_zeros=True)
    print("Number of nuclei after filtering:", n_ids)

    return seg_filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder", required=True)
    args = parser.parse_args()

    seg_path = os.path.join(args.output_folder, "segmentation.zarr")
    seg_key = "segmentation"

    seg_filtered = filter_segmentation(seg_path, seg_key)

    with z5py.File(seg_path, "a") as f:
        chunks = f[seg_key].chunks
        f.create_dataset(
            "segmentation_postprocessed", data=seg_filtered, compression="gzip",
            chunks=chunks, dtype=seg_filtered.dtype
        )


if __name__ == "__main__":
    main()
