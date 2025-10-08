import os
import multiprocessing as mp
from concurrent import futures
from typing import List, Tuple

import numpy as np
import tifffile
from tqdm import tqdm


def coord_from_string(center_str):
    return tuple([int(c) for c in center_str.split("-")])


def find_annotations(annotation_dir: str, cochlea: str, pattern: str = None) -> dict:
    """Create a dictionary for the analysis of ChReef annotations.

    Annotations should have format positive-negative_<cochlea>_crop_<coord>_allNegativeExcluded_thr<thr>.tif

    Args:
        annotation_dir: Directory containing annotations.
        cochlea: The name of the cochlea to analyze.

    Returns:
        Dictionary with information about the intensity annotations.
    """

    def extract_center_string(cochlea, name):
        # Extract center crop coordinate from file name
        crop_suffix = name.split(f"{cochlea}_crop_")[1]
        center_str = crop_suffix.split("_")[0]
        return center_str

    if pattern is not None:
        cochlea_files = [entry.name for entry in os.scandir(annotation_dir) if cochlea in entry.name
                         and pattern in entry.name]
    else:
        cochlea_files = [entry.name for entry in os.scandir(annotation_dir) if cochlea in entry.name]
    dic = {"cochlea": cochlea}
    dic["cochlea_files"] = cochlea_files
    center_strings = list(set([extract_center_string(cochlea, name=f) for f in cochlea_files]))
    center_strings.sort()
    dic["center_strings"] = center_strings
    remove_strings = []
    for center_str in center_strings:
        files_neg = [c for c in cochlea_files if all(x in c for x in [cochlea, center_str, "NegativeExcluded"])]
        files_pos = [c for c in cochlea_files if all(x in c for x in [cochlea, center_str, "WeakPositive"])]
        if len(files_neg) != 1 or len(files_pos) != 1:
            print(f"Skipping crop {center_str} for cochlea {cochlea}. "
                  f"Missing or multiple annotation files in {annotation_dir}.")
            remove_strings.append(center_str)
        else:
            dic[center_str] = {"file_neg": os.path.join(annotation_dir, files_neg[0]),
                               "file_pos": os.path.join(annotation_dir, files_pos[0])}
    for rm_str in remove_strings:
        dic["center_strings"].remove(rm_str)

    return dic


def get_roi(coord: tuple, roi_halo: tuple, resolution: float = 0.38) -> Tuple[int]:
    """Get parameters for loading ROI of segmentation.

    Args:
        coord: Center coordinate.
        roi_halo: Halo for roi.
        resolution: Resolution of array in µm.

    Returns:
        The region of interest.
    """
    coords = list(coord)
    # reverse dimensions for correct extraction
    coords.reverse()
    coords = np.array(coords)
    if not isinstance(resolution, float):
        assert len(resolution) == 3
        resolution = np.array(resolution)[::-1]
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
    print(f"Finding overlapping masks with {n_threads} Threads.")
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
        roi_seg: Region of interest of segmentation.

    Returns:
        A list of the ids that are in between the respective thresholds.
    """
    # negative annotation == 1, positive annotation == 2
    negexc_negatives = find_overlapping_masks(arr_negexc, roi_seg, label_id_base=1)
    allweak_positives = find_overlapping_masks(arr_allweak, roi_seg, label_id_base=2)
    inbetween_ids = [int(i) for i in set(negexc_negatives).intersection(set(allweak_positives))]
    return inbetween_ids, allweak_positives, negexc_negatives


def get_median_intensity(file_negexc, file_allweak, center, data_seg, table, column="median",
                         resolution=0.38):
    arr_negexc = tifffile.imread(file_negexc)
    arr_allweak = tifffile.imread(file_allweak)

    roi_halo = tuple([r // 2 for r in arr_negexc.shape])
    roi = get_roi(center, roi_halo, resolution=resolution)

    roi_seg = data_seg[roi]
    inbetween_ids, allweak_positives, negexc_negatives = find_inbetween_ids(arr_negexc, arr_allweak, roi_seg)
    if len(inbetween_ids) == 0:
        if len(allweak_positives) == 0 and len(negexc_negatives) == 0:
            return None

        subset_positive = table[table["label_id"].isin(allweak_positives)]
        subset_negative = table[table["label_id"].isin(negexc_negatives)]
        lowest_positive = float(subset_positive[column].min())
        highest_negative = float(subset_negative[column].max())
        if np.isnan(lowest_positive) or np.isnan(highest_negative):
            return None

        return np.average([lowest_positive, highest_negative])

    subset = table[table["label_id"].isin(inbetween_ids)]
    intensities = list(subset[column])

    return np.median(list(intensities))


def localize_median_intensities(annotation_dir, cochlea, data_seg, table_measure, column="median", pattern=None,
                                resolution=0.38):
    """Find median intensities in blocks and assign them to center positions of cropped block.
    """
    annotation_dic = find_annotations(annotation_dir, cochlea, pattern=pattern)
    # center_keys = [key for key in annotation_dic["center_strings"] if key in annotation_dic.keys()]

    for center_str in annotation_dic["center_strings"]:
        center_coord = coord_from_string(center_str)
        print(f"Getting median intensities for {center_coord}.")
        file_pos = annotation_dic[center_str]["file_pos"]
        file_neg = annotation_dic[center_str]["file_neg"]
        median_intensity = get_median_intensity(file_neg, file_pos, center_coord, data_seg,
                                                table_measure, column=column, resolution=resolution)

        if median_intensity is None:
            print(f"No threshold identified for {center_str}.")

        annotation_dic[center_str]["median_intensity"] = median_intensity

    return annotation_dic
