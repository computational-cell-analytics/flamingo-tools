import os
from glob import glob
from pathlib import Path

import pandas as pd
from elf.io import open_file

from flamingo_tools.validation import match_detections


def evaluate_synapse_detections(pred, gt):
    fname = os.path.basename(gt)

    pred = pd.read_csv(pred, sep="\t")[["z", "y", "x"]].values
    gt = pd.read_csv(gt, sep="\t")[["z", "y", "x"]].values
    tps_pred, tps_gt, fps, fns = match_detections(pred, gt, max_dist=3)

    return pd.DataFrame({
        "name": [fname], "tp": [len(tps_pred)], "fp": [len(fps)], "fn": [len(fns)],
    })


def run_evaluation(pred_files, gt_files):
    results = []
    for pred, gt in zip(pred_files, gt_files):
        res = evaluate_synapse_detections(pred, gt)
        results.append(res)
    results = pd.concat(results)

    tp = results.tp.sum()
    fp = results.fp.sum()
    fn = results.fn.sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print("All results:")
    print(results)
    print("Evaluation:")
    print("Precision:", precision)
    print("Revall:", recall)
    print("F1-Score:", f1_score)


def visualize_synapse_detections(pred, gt, heatmap_path=None, ctbp2_path=None):
    import napari

    fname = os.path.basename(gt)

    pred = pd.read_csv(pred, sep="\t")[["z", "y", "x"]].values
    gt = pd.read_csv(gt, sep="\t")[["z", "y", "x"]].values
    tps_pred, tps_gt, fps, fns = match_detections(pred, gt, max_dist=4)

    tps = pred[tps_pred]
    fps = pred[fps]
    fns = gt[fns]

    if heatmap_path is None:
        heatmap = None
    else:
        heatmap = open_file(heatmap_path)["prediction"][:]

    if ctbp2_path is None:
        ctbp2 = None
    else:
        ctbp2 = open_file(ctbp2_path)["raw"][:]

    v = napari.Viewer()
    if ctbp2 is not None:
        v.add_image(ctbp2)
    if heatmap is not None:
        v.add_image(heatmap)
    v.add_points(pred, visible=False)
    v.add_points(gt, visible=False)
    v.add_points(tps, name="TPS", face_color="green")
    v.add_points(fps, name="FPs", face_color="orange")
    v.add_points(fns, name="FNs", face_color="yellow")
    v.title = f"{fname}: tps={len(tps)}, fps={len(fps)}, fns={len(fns)}"
    napari.run()


def visualize_evaluation(pred_files, gt_files, ctbp2_files):
    for pred, gt, ctbp2 in zip(pred_files, gt_files, ctbp2_files):
        pred_folder = os.path.split(pred)[0]
        heatmap = os.path.join(pred_folder, "predictions.zarr")
        visualize_synapse_detections(pred, gt, heatmap, ctbp2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    pred_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/SynapseValidation"
    gt_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v3/labels"
    ctbp2_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v3/images"  # noqa

    ctbp2_files = sorted(glob(os.path.join(ctbp2_root, "*.zarr")))
    gt_files = sorted(glob(os.path.join(gt_root, "*_filtered.tsv")))
    assert len(ctbp2_files) == len(gt_files)

    pred_files = []
    for ff in ctbp2_files:
        fname = Path(ff).stem
        pred_file = os.path.join(pred_root, fname, "filtered_synapse_detection.tsv")
        assert os.path.exists(pred_file), pred_file
        pred_files.append(pred_file)

    if args.visualize:
        visualize_evaluation(pred_files, gt_files, ctbp2_files)
    else:
        run_evaluation(pred_files, gt_files)


if __name__ == "__main__":
    main()
