import os
import multiprocessing as mp
from concurrent import futures
from typing import List, Tuple

import numpy as np
import tifffile
from tqdm import tqdm


def find_annotations(annotation_dir) -> dict:
    """Create dictionary for analysis of ChReef annotations.
    Annotations should have format positive-negative_<cochlea>_crop_<coord>_allNegativeExcluded_thr<thr>.tif

    Args:
        annotation_dir: Directory containing annotations.
    """

    def extract_center_crop(cochlea, name):
        # Extract center crop coordinate from file name
        crop_suffix = name.split(f"{cochlea}_crop_")[1]
        coord_str = crop_suffix.split("_")[0]
        coord = tuple([int(c) for c in coord_str.split("-")])
        return coord

    def extract_cochlea_str(name):
        # Extract cochlea str from annotation file name.
        cochlea_suffix = name.split("negative_")[1]
        cochlea = cochlea_suffix.split("_crop")[0]
        return cochlea

    file_names = [entry.name for entry in os.scandir(annotation_dir)]
    cochleae = list(set([extract_cochlea_str(file_name) for file_name in file_names]))
    annotation_dic = {}
    for cochlea in cochleae:
        cochlea_files = [entry.name for entry in os.scandir(annotation_dir) if cochlea in entry.name]
        dic = {"cochlea": cochlea}
        dic["cochlea_files"] = cochlea_files
        center_crops = list(set([extract_center_crop(cochlea, name=file_name) for file_name in cochlea_files]))
        dic["center_coords"] = center_crops
        dic["center_coords_str"] = [("-").join([str(c).zfill(4) for center_crop in center_crops for c in center_crop])]
        for center_str in dic["center_coords_str"]:
            file_neg = [c for c in cochlea_files if all(x in c for x in [cochlea, center_str, "NegativeExcluded"])][0]
            file_pos = [c for c in cochlea_files if all(x in c for x in [cochlea, center_str, "WeakPositive"])][0]
            dic[center_str] = {"file_neg": file_neg, "file_pos": file_pos}
        annotation_dic[cochlea] = dic
    return annotation_dic


def get_roi(coord: tuple, roi_halo: tuple, resolution: float = 0.38) -> Tuple[int]:
    """Get parameters for loading ROI of segmentation.

    Args:
        coord: Center coordinate.
        roi_halo: Halo for roi.
        resolution: Resolution of array in µm.

    Returns:
        region of interest
    """
    coords = list(coord)
    # reverse dimensions for correct extraction
    coords.reverse()
    coords = np.array(coords)
    coords = coords / resolution
    coords = np.round(coords).astype(np.int32)

    roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coords, roi_halo))
    return roi


def find_overlapping_masks(
    arr_base: np.ndarray,
    arr_ref: np.ndarray,
    label_id_base: int,
    min_overlap: float = 0.5,
) -> List[int]:
    """Find overlapping masks between base array and reference array.

    Args:
        arr_base: Base array.
        arr_ref: Reference array.
        label_id_base: ID of segmentation to check for overlap.
        min_overlap: Minimal overlap to consider segmentation ID as matching.

    Returns:
        Matching IDs of reference array.
    """
    arr_base_labeled = arr_base == label_id_base

    # iterate through segmentation ids in reference mask
    ref_ids = list(np.unique(arr_ref)[1:])

    def check_overlap(ref_id):
        # check overlap of reference ID and base
        arr_ref_instance = arr_ref == ref_id

        intersection = np.logical_and(arr_ref_instance, arr_base_labeled)
        overlap_ratio = np.sum(intersection) / np.sum(arr_ref_instance)
        if overlap_ratio >= min_overlap:
            return ref_id
        else:
            return None

    n_threads = min(16, mp.cpu_count())
    print(f"Parallelizing with {n_threads} Threads.")
    with futures.ThreadPoolExecutor(n_threads) as pool:
        results = list(tqdm(pool.map(check_overlap, ref_ids), total=len(ref_ids)))

    matching_ids = {r for r in results if r is not None}
    return matching_ids


def find_inbetween_ids(
    arr_negexc: np.typing.ArrayLike,
    arr_allweak: np.typing.ArrayLike,
    roi_seg: np.typing.ArrayLike,
) -> List[int]:
    """Identify list of segmentation IDs inbetween thresholds.

    Args:
        arr_negexc: Array with all negatives excluded.
        arr_allweak: Array with all weak positives.
        roi_sgn: Region of interest of segmentation.
    """
    # negative annotation == 1, positive annotation == 2
    negexc_negatives = find_overlapping_masks(arr_negexc, roi_seg, label_id_base=1)
    allweak_positives = find_overlapping_masks(arr_allweak, roi_seg, label_id_base=2)
    inbetween_ids = list(set(negexc_negatives) & set(allweak_positives))
    return inbetween_ids


def get_median_intensity(file_negexc, file_allweak, center, data_seg, table):
    arr_negexc = tifffile.imread(file_negexc)
    arr_allweak = tifffile.imread(file_allweak)

    roi_halo = tuple([r // 2 for r in arr_negexc.shape])
    roi = get_roi(center, roi_halo)

    roi_seg = data_seg[roi]
    inbetween_ids = find_inbetween_ids(arr_negexc, arr_allweak, roi_seg)
    intensities = table.loc[table["label_id"].isin(inbetween_ids), table["mean"]]
    return np.median(list(intensities))
