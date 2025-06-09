from typing import Optional

import numpy as np
import pandas as pd

from elf.parallel.distance_transform import map_points_to_objects


def map_and_filter_detections(
    segmentation: np.ndarray,
    detections: pd.DataFrame,
    max_distance: float,
    resolution: float = 0.38,
    n_threads: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Map synapse detections to segmented IHCs and filter out detections above a distance threshold to the IHCs.

    Args:
        segmentation: The IHC segmentation.
        detections: The synapse marker detections.
        max_distance: The maximal distance for a valid match of synapse markers to IHCs.
        resolution: The resolution / voxel size of the data in micrometer.
        n_threads: The number of threads for parallelizing the mapping of detections to objects.
        verbose: Whether to print the progress of the mapping procedure.

    Returns:
        The filtered dataframe with the detections mapped to the segmentation.
    """
    # Get the point coordinates.
    points = detections[["z", "y", "x"]].values.astype("int")

    # Set the block shape (this could also be exposed as a parameter; it should not matter much though).
    block_shape = (64, 256, 256)

    # Determine the halo. We set it to 2 pixels + the max-distance in pixels, to ensure all distances
    # that are smaller than the max distance are measured.
    halo = (2 + int(np.ceil(max_distance / resolution)),) * 3

    # Map the detections to the obejcts in the (IHC) segmentation.
    object_ids, object_distances = map_points_to_objects(
        segmentation=segmentation,
        points=points,
        block_shape=block_shape,
        halo=halo,
        sampling=resolution,
        n_threads=n_threads,
        verbose=verbose,
    )
    assert len(object_ids) == len(points)
    assert len(object_distances) == len(points)

    # Add matched ids and distances to the dataframe.
    detections["matched_ihc"] = object_ids
    detections["distance_to_ihc"] = object_distances

    # Filter the dataframe by the max distance.
    detections = detections[detections.distance_to_ihc < max_distance]
    return detections


# TODO implement streamlined workflow for the marker detection, mapping and filtering.
def marker_detection():
    """
    """

    # 1.) Determine mask for inference based on the IHC segmentation.
    # Best approach: load IHC segmentation at a low scale level, binarize it,
    # dilate it and use this as mask. It can be mapped back to the full resolution
    # with `elf.wrapper.ResizedVolume`.

    # 2.) Run inference and detection of maxima.
    # This can be taken from 'scripts/synapse_marker_detection/run_prediction.py'
    # (And the run prediction script should then be refactored).

    # 3.) Map the detections to IHC and filter them based on a distance criterion.
    # Use the function 'map_and_filter_detections' from above.

    # 4.) Add the filtered detections to MoBIE.
    # IMPORTANT scale the coordinates with the resolution here.
