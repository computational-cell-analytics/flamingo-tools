import os
import re
from typing import Dict, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import zarr

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


# For a less naive annotation we may need to also fetch +- a few slices,
# so that we have a bit of tolerance with the distance based matching.
def fetch_data_for_evaluation(
    annotation_path: str,
    cache_path: Optional[str] = None,
    seg_name: str = "SGN",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    """
    annotations = pd.read_csv(annotation_path)
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

    # Download the segmentation for this slice.
    input_key = "s0"
    with zarr.open(s3_store, mode="r") as f:
        segmentation = f[input_key][slice_id]

    # Cache it if required.
    if cache_path is not None:
        imageio.imwrite(cache_path, segmentation, compression="zlib")

    return segmentation, annotations


def evaluate_annotated_slice(
    segmentation: np.typing.ArrayLike,
    annotations: pd.DataFrame,
    matching_tolerance: float = 0.0,
) -> Dict[str, int]:
    """Computes the scores for a annotated validation slice.

    Computes true positives, false positives and false negatives for scoring.

    Args:
        segmentation: The segmentation for this slide.
        annotations: The annotations, marking cell centers.
        matching_tolerance: ...

    Returns:
        A dictionary with keys 'tp', 'fp' and 'fn', mapping to the respective counts.
    """
    # Compute the distance transform and nearest id fields.

    # Match all of the points to segmented objects based on their distance.

    # Determine the TPs, FPs and FNs based on a linear cost assignment.
    tp = ...
    fp = ...
    fn = ...

    return {"tp": tp, "fp": fp, "fn": fn}
