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
                                  "final_annotations",
                                  "final_consensus_annotations")

    baselines = [
        "cellpose3",
        "cellpose-sam",
        "distance_unet",
        "micro-sam",
        "stardist"]

    for baseline in baselines:
        eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)


def eval_all_ihc():
    """Evaluate all IHC baselines.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_ihc")
    annotation_dir = (cochlea_dir, "AnnotatedImageCrops/F1ValidationIHCs/consensus_annotation")
    baselines = [
        "cellpose3",
        "cellpose-sam",
        "distance_unet_v3",
        "micro-sam"]

    for baseline in baselines:
        eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)


def eval_segmentation(seg_dir, annotation_dir):
    print(f"Evaluating segmentation in directory {seg_dir}")
    segs = [entry.path for entry in os.scandir(seg_dir) if entry.is_file() and ".tif" in entry.path]

    seg_dicts = []
    for seg in segs:

        basename = os.path.basename(seg)
        basename = ".".join(basename.split(".")[:-1])
        basename = "".join(basename.split("_seg")[0])
        print(basename)
        print("Annotation_dir", annotation_dir)
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
                                                          annotations=df,
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


def print_accuracy(eval_dir):
    """Print 'Precision', 'Recall', and 'F1-score' for dictionaries in a given directory.
    """
    eval_dicts = [entry.path for entry in os.scandir(eval_dir) if entry.is_file() and "dic.json" in entry.path]
    precision_list = []
    recall_list = []
    f1_score_list = []
    for eval_dic in eval_dicts:
        with open(eval_dic, "r") as f:
            d = json.load(f)
        tp = len(d["tp_objects"])
        fp = len(d["fp"])
        fn = len(d["fn"])

        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        if precision + recall != 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else: f1_score = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    names = ["Precision", "Recall", "F1 score"]
    for num, lis in enumerate([precision_list, recall_list, f1_score_list]):
        print(names[num], sum(lis) / len(lis))


def print_accuracy_sgn():
    """Print 'Precision', 'Recall', and 'F1-score' for all SGN baselines.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_sgn")
    baselines = [
        "cellpose3",
        "cellpose-sam",
        "distance_unet",
        "micro-sam",
        "stardist"]
    for baseline in baselines:
        print(f"Evaluating baseline {baseline}")
        print_accuracy(os.path.join(seg_dir, baseline))


def print_accuracy_ihc():
    """Print 'Precision', 'Recall', and 'F1-score' for all IHC baselines.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_ihc")
    baselines = [
        "cellpose3",
        "cellpose-sam",
        "distance_unet_v3",
        "micro-sam"]

    for baseline in baselines:
        print(f"Evaluating baseline {baseline}")
        print_accuracy(os.path.join(seg_dir, baseline))


def main():
    #eval_all_sgn()
    #eval_all_ihc()
    #print_accuracy_sgn()
    print_accuracy_ihc()


if __name__ == "__main__":
    main()
