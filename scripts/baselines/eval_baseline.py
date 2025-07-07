import os
import json

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from flamingo_tools.validation import compute_matches_for_annotated_slice


def filter_seg(seg_arr: np.typing.ArrayLike, min_count: int = 3000, max_count: int = 50000):
    """Filter segmentation based on minimal and maximal number of pixels of the segmented object.

    Args:
        seg_arr: Input segmentation
        min_count: Minimal number of pixels
        max_count: Maximal number of pixels

    Returns:
        Filtered segmentation
    """
    segmentation_ids = np.unique(seg_arr)[1:]
    seg_counts = [np.count_nonzero(seg_arr == seg_id) for seg_id in segmentation_ids]
    seg_filtered = [idx for idx, seg_count in zip(segmentation_ids, seg_counts) if min_count <= seg_count <= max_count]
    for s in segmentation_ids:
        if s not in seg_filtered:
            seg_arr[seg_arr == s] = 0
    return seg_arr


def eval_seg_dict(dic: dict, out_path: str):
    """Format dictionary entries and write dictionary to output path.

    Args:
        dic: Parameter dictionary for baseline evaluation.
        out_path: Output path for json file.
    """
    dic["tp_objects"] = list(dic["tp_objects"])
    dic["tp_annotations"] = list(dic["tp_annotations"])
    dic["fp"] = list(dic["fp"])
    dic["fn"] = list(dic["fn"])

    dic["tp_objects"] = [int(i) for i in dic["tp_objects"]]
    dic["tp_annotations"] = [int(i) for i in dic["tp_annotations"]]
    dic["fp"] = [int(i) for i in dic["fp"]]
    dic["fn"] = [int(i) for i in dic["fn"]]

    json_out = os.path.join(out_path)
    with open(json_out, "w") as f:
        json.dump(dic, f, indent='\t', separators=(',', ': '))


def eval_all_sgn():
    """Evaluate all SGN baselines.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_sgn")
    annotation_dir = os.path.join(cochlea_dir,
                                  "AnnotatedImageCrops",
                                  "F1ValidationSGNs",
                                  "for_consensus_annotation",
                                  "consensus_annotations")
    baselines = [
        "cellpose3",
        "cellpose-sam"
        "distance-unet",
        "micro-sam",
        "stardist"]

    for baseline in baselines:
        eval_segmentation(seg_dir=seg_dir, annotation_dir=annotation_dir)


def eval_all_ihc():
    """Evaluate all IHC baselines.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_ihc")
    annotation_dir = (cochlea_dir, "AnnotatedImageCrops/F1ValidationIHCs/consensus_annotation")
    baselines = [
        "cellpose3",
        "cellpose-sam"
        "distance-unet_v3",
        "micro-sam"]

    for baseline in baselines:
        eval_segmentation(seg_dir=seg_dir, annotation_dir=annotation_dir)


def eval_segmentation(seg_dir, annotation_dir):
    segs = [entry.path for entry in os.scandir(seg_dir) if entry.is_file() and ".tif" in entry.path]

    seg_dicts = []
    for seg in segs:

        basename = os.path.basename(seg)
        basename = ".".join(basename.split(".")[:-1])
        basename = "".join(basename.split("_seg")[0])
        print(basename)
        dic_out = os.path.join(seg_dir, f"{basename}_dic.json")
        if not os.path.isfile(dic_out):

            df_path = os.path.join(annotation_dir, f"{basename}.csv")
            df = pd.read_csv(df_path, sep=",")
            timer_file = os.path.join(seg_dir, f"{basename}_timer.json")
            with open(timer_file) as f:
                timer_dic = json.load(f)

            seg_arr = imageio.imread(seg)
            seg_filtered = filter_seg(seg_arr=seg_arr)

            seg_dic = compute_matches_for_annotated_slice(segmentation=seg_filtered,
                                                          annotations=annotation_dir,
                                                          matching_tolerance=5)
            seg_dic["annotation_length"] = len(df)
            seg_dic["crop_name"] = basename
            seg_dic["time"] = float(timer_dic["total_duration[s]"])

            eval_seg_dict(seg_dic, dic_out)

            seg_dicts.append(seg_dic)
        else:
            print(f"Dictionary {dic_out} already exists")

    json_out = os.path.join(seg_dir, "eval_seg.json")
    with open(json_out, "w") as f:
        json.dump(seg_dicts, f, indent='\t', separators=(',', ': '))
