import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
from skimage.measure import label
from skimage.segmentation import relabel_sequential

from torch_em.util import load_model
from torch_em.util.grid_search import DistanceBasedInstanceSegmentation, instance_segmentation_grid_search

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/IHC/2025-07-for-grid-search"
MODEL_PATH = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/v4_cochlea_distance_unet_IHC_supervised_2025-07-14"  # noqa

GRID_SEARCH_VALUES = {
    "center_distance_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "boundary_distance_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "distance_smoothing": [0.0, 0.6, 1.0, 1.6],
    "min_size": [0],
}


def preprocess_gt():
    label_paths = sorted(glob(os.path.join(ROOT, "**/*_corrected.tif"), recursive=True))
    for label_path in label_paths:
        seg = imageio.imread(label_path)
        seg = label(seg)

        min_size = 750
        ids, sizes = np.unique(seg, return_counts=True)
        filter_ids = ids[sizes < min_size]
        seg[np.isin(seg, filter_ids)] = 0
        seg, _, _ = relabel_sequential(seg)

        out_path = label_path.replace("_corrected", "_gt")
        imageio.imwrite(out_path, seg)


def run_grid_search():
    image_paths = sorted(glob(os.path.join(ROOT, "**/*lut3*.tif"), recursive=True))
    label_paths = sorted(glob(os.path.join(ROOT, "**/*_gt.tif"), recursive=True))
    assert len(image_paths) == len(label_paths), f"{len(image_paths)}, {len(label_paths)}"
    result_dir = "ihc-v4-gs"

    block_shape = (96, 256, 256)
    halo = (8, 64, 64)
    model = load_model(MODEL_PATH)
    segmenter = DistanceBasedInstanceSegmentation(model, block_shape=block_shape, halo=halo)

    best_kwargs, best_score = instance_segmentation_grid_search(
        segmenter, image_paths, label_paths, result_dir, grid_search_values=GRID_SEARCH_VALUES
    )
    print("Grid-search result:")
    print(best_kwargs)
    print(best_score)


# TODO plot the grid search results
def evaluate_grid_search():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    result_dir = "ihc-v4-gs"
    criterion = "SA50"

    gs_files = glob(os.path.join(result_dir, "*.csv"))
    gs_result = pd.concat([pd.read_csv(gs_file) for gs_file in gs_files])

    grid_search_parameters = list(GRID_SEARCH_VALUES.keys())

    # Retrieve only the relevant columns and group by the gridsearch columns
    # and compute the mean value
    gs_result = gs_result[grid_search_parameters + [criterion]].reset_index()
    gs_result = gs_result.groupby(grid_search_parameters).mean().reset_index()

    # Find the best score and best result
    best_score, best_idx = gs_result[criterion].max(), gs_result[criterion].idxmax()
    best_params = gs_result.iloc[best_idx]
    best_kwargs = {k: v for k, v in zip(grid_search_parameters, best_params)}

    print("Best parameters:")
    print(best_kwargs)
    print("With score:", best_score)

    fig, axes = plt.subplots(3)
    for i, (idx, col) in enumerate([
        ("center_distance_threshold", "boundary_distance_threshold"),
        ("distance_smoothing", "boundary_distance_threshold"),
        ("distance_smoothing", "center_distance_threshold"),
    ]):
        res = gs_result.groupby([idx, col]).mean().reset_index()
        res = res[[idx, col, criterion]]
        res = res.pivot(index=idx, columns=col, values=criterion)
        sns.heatmap(res, cmap="viridis", annot=True, fmt=".2g", cbar_kws={"label": criterion}, ax=axes[i])
        axes[i].set_xlabel(idx)
        axes[i].set_xlabel(col)
    plt.show()


def main():
    # preprocess_gt()
    # run_grid_search()
    evaluate_grid_search()


if __name__ == "__main__":
    main()
