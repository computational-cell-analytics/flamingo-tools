import os
import re
from typing import Dict, List, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import zarr

from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

from .s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT


def _normalize_cochlea_name(name):
    match = re.search(r"\d+", name)
    pos = match.start() if match else None
    assert pos is not None, name
    prefix = name[:pos]
    prefix = f"{prefix[0]}_{prefix[1:]}"
    number = int(name[pos:-1])
    postfix = name[-1]
    return f"{prefix}_{number:06d}_{postfix}"


# TODO enable table component filtering with MoBIE table
# NOTE: the main component is always #1
def fetch_data_for_evaluation(
    annotation_path: str,
    cache_path: Optional[str] = None,
    seg_name: str = "SGN",
    z_extent: int = 0,
    components_for_postprocessing: Optional[List[int]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    """
    # Load the annotations and normalize them for the given z-extent.
    annotations = pd.read_csv(annotation_path)
    annotations = annotations.drop(columns="index")
    if z_extent == 0:  # If we don't have a z-extent then we just drop the first axis and rename the other two.
        annotations = annotations.drop(columns="axis-0")
        annotations = annotations.rename(columns={"axis-1": "axis-0", "axis-2": "axis-1"})
    else:  # Otherwise we have to center the first axis.
        # TODO
        raise NotImplementedError

    # Load the segmentaiton from cache path if it is given and if it is already cached.
    if cache_path is not None and os.path.exists(cache_path):
        segmentation = imageio.imread(cache_path)
        return segmentation, annotations

    # Parse which ID and which cochlea from the name.
    fname = os.path.basename(annotation_path)
    name_parts = fname.split("_")
    cochlea = _normalize_cochlea_name(name_parts[0])
    slice_id = int(name_parts[2][1:])

    # Open the S3 connection, get the path to the SGN segmentation in S3.
    internal_path = os.path.join(cochlea, "images",  "ome-zarr", f"{seg_name}.ome.zarr")
    s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)

    # Compute the roi for the given z-extent.
    if z_extent == 0:
        roi = slice_id
    else:
        roi = slice(slice_id - z_extent, slice_id + z_extent)

    # Download the segmentation for this slice and the given z-extent.
    input_key = "s0"
    with zarr.open(s3_store, mode="r") as f:
        segmentation = f[input_key][roi]

    if components_for_postprocessing is not None:
        # Filter the IDs so that only the ones part of 'components_for_postprocessing_remain'.

        # First, we download the MoBIE table for this segmentation.
        internal_path = os.path.join(BUCKET_NAME, cochlea, "tables",  seg_name, "default.tsv")
        with fs.open(internal_path, "r") as f:
            table = pd.read_csv(f, sep="\t")

        # Then we get the ids for the components and us them to filter the segmentation.
        component_mask = np.isin(table.component_labels.values, components_for_postprocessing)
        keep_label_ids = table.label_id.values[component_mask].astype("int64")
        filter_mask = ~np.isin(segmentation, keep_label_ids)
        segmentation[filter_mask] = 0

    segmentation, _, _ = relabel_sequential(segmentation)

    # Cache it if required.
    if cache_path is not None:
        imageio.imwrite(cache_path, segmentation, compression="zlib")

    return segmentation, annotations


# TODO crop to the bounding box around the union of points and segmentation masks to be more efficient.
def compute_matches_for_annotated_slice(
    segmentation: np.typing.ArrayLike,
    annotations: pd.DataFrame,
    matching_tolerance: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Computes the ids of matches and non-matches for a annotated validation slice.

    Computes true positive ids (for objects and annotations), false positive ids and false negative ids
    by solving a linear cost assignment of distances between objects and annotations.

    Args:
        segmentation: The segmentation for this slide. We assume that it is relabeled consecutively.
        annotations: The annotations, marking cell centers.
        matching_tolerance: The maximum distance for matching an annotation to a segmented object.

    Returns:
        A dictionary with keys 'tp_objects', 'tp_annotations' 'fp' and 'fn', mapping to the respective ids.
    """
    assert segmentation.ndim in (2, 3)
    segmentation_ids = np.unique(segmentation)[1:]
    n_objects, n_annotations = len(segmentation_ids), len(annotations)

    # In order to get the full distance matrix, we compute the distance to all objects for each annotation.
    # This is not very efficient, but it's the most straight-forward and most rigorous approach.
    scores = np.zeros((n_objects, n_annotations), dtype="float")
    coordinates = ["axis-0", "axis-1"] if segmentation.ndim == 2 else ["axis-0", "axis-1", "axis-2"]
    for i, row in tqdm(annotations.iterrows(), total=n_annotations, desc="Compute pairwise distances"):
        coordinate = tuple(int(np.round(row[coord])) for coord in coordinates)
        distance_input = np.ones(segmentation.shape, dtype="bool")
        distance_input[coordinate] = False
        distances, indices = distance_transform_edt(distance_input, return_indices=True)

        props = regionprops_table(segmentation, intensity_image=distances, properties=("label", "min_intensity"))
        distances = props["min_intensity"]
        assert len(distances) == scores.shape[0]
        scores[:, i] = distances

    # Find the assignment of points to objects.
    # These correspond to the TP ids in the point / object annotations.
    tp_ids_objects, tp_ids_annotations = linear_sum_assignment(scores)
    match_ok = scores[tp_ids_objects, tp_ids_annotations] <= matching_tolerance
    tp_ids_objects, tp_ids_annotations = tp_ids_objects[match_ok], tp_ids_annotations[match_ok]
    tp_ids_objects = segmentation_ids[tp_ids_objects]
    assert len(tp_ids_objects) == len(tp_ids_annotations)

    # Find the false positives: objects that are not part of the matches.
    fp_ids = np.setdiff1d(segmentation_ids, tp_ids_objects)

    # Find the false negatives: annotations that are not part of the matches.
    fn_ids = np.setdiff1d(np.arange(n_annotations), tp_ids_annotations)

    return {"tp_objects": tp_ids_objects, "tp_annotations": tp_ids_annotations, "fp": fp_ids, "fn": fn_ids}


def compute_scores_for_annotated_slice(
    segmentation: np.typing.ArrayLike,
    annotations: pd.DataFrame,
    matching_tolerance: float = 0.0,
) -> Dict[str, int]:
    """Computes the scores for a annotated validation slice.

    Computes true positives, false positives and false negatives for scoring.

    Args:
        segmentation: The segmentation for this slide. We assume that it is relabeled consecutively.
        annotations: The annotations, marking cell centers.
        matching_tolerance: The maximum distance for matching an annotation to a segmented object.

    Returns:
        A dictionary with keys 'tp', 'fp' and 'fn', mapping to the respective counts.
    """
    result = compute_matches_for_annotated_slice(segmentation, annotations, matching_tolerance)

    # To determine the TPs, FPs and FNs.
    tp = len(result["tp_objects"])
    fp = len(result["fp"])
    fn = len(result["fn"])
    return {"tp": tp, "fp": fp, "fn": fn}


def for_visualization(segmentation, annotations, matches):
    green_red = ["#00FF00", "#FF0000"]

    seg_vis = np.zeros_like(segmentation)
    tps, fps = matches["tp_objects"], matches["fp"]
    seg_vis[np.isin(segmentation, tps)] = 1
    seg_vis[np.isin(segmentation, fps)] = 2

    # TODO red / green colormap
    seg_props = dict(color={1: green_red[0], 2: green_red[1]})

    point_vis = annotations.copy()
    tps = matches["tp_annotations"]
    point_props = dict(
        properties={"match": [0 if aid in tps else 1 for aid in range(len(annotations))]},
        border_color="match",
        border_color_cycle=green_red,
    )

    return seg_vis, point_vis, seg_props, point_props
