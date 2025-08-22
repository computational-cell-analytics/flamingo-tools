import os
from glob import glob
from pathlib import Path
import json

import imageio.v3 as imageio
import napari
import numpy as np
from skimage.segmentation import find_boundaries

FOR_COMPARISON = ["distance_unet", "micro-sam", "cellpose3"]


def _eval_seg(seg, eval_path):
    with open(eval_path, "r") as f:
        eval_res = json.load(f)

    correct, wrong = eval_res["tp_objects"], eval_res["fp"]
    all_ids = correct + wrong
    seg[~np.isin(seg, all_ids)] = 0

    eva_mask = np.zeros_like(seg)

    eva_mask[np.isin(seg, correct)] = 1
    eva_mask[np.isin(seg, wrong)] = 2

    bd = find_boundaries(seg)
    return bd, eva_mask


def sgn_comparison():
    z = 10

    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    val_sgn_dir = f"{cochlea_dir}/predictions/val_sgn"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation"

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))

    for path in image_paths:
        image = imageio.imread(path)[z]

        seg_fname = Path(path).stem + "_seg.tif"
        eval_fname = Path(path).stem + "_dic.json"

        segmentations, boundaries, eval_im = {}, {}, {}
        for seg_name in FOR_COMPARISON:
            seg_path = os.path.join(val_sgn_dir, seg_name, seg_fname)
            eval_path = os.path.join(val_sgn_dir, seg_name, eval_fname)
            assert os.path.exists(seg_path), seg_path

            seg = imageio.imread(seg_path)[z]

            bd, eva = _eval_seg(seg, eval_path)
            segmentations[seg_name] = seg
            boundaries[seg_name] = bd
            eval_im[seg_name] = eva

        v = napari.Viewer()
        v.add_image(image)
        for seg_name, bd in boundaries.items():
            v.add_labels(bd, name=seg_name, colormap={1: "cyan"})
            v.add_labels(eval_im[seg_name], name=f"{seg_name}_eval", colormap={1: "green", 2: "red"})
        v.title = Path(path).stem
        napari.run()


def ihc_comparison():
    z = 10

    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    val_sgn_dir = f"{cochlea_dir}/predictions/val_ihc"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationIHCs"

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))

    for path in image_paths:
        image = imageio.imread(path)[z]

        seg_fname = Path(path).stem + "_seg.tif"
        eval_fname = Path(path).stem + "_dic.json"

        segmentations, boundaries, eval_im = {}, {}, {}
        for seg_name in FOR_COMPARISON:
            # FIXME distance_unet_v4b is missing the eval files
            seg_name_ = "distance_unet_v3" if seg_name == "distance_unet" else seg_name
            seg_path = os.path.join(val_sgn_dir, seg_name_, seg_fname)
            eval_path = os.path.join(val_sgn_dir, seg_name_, eval_fname)
            assert os.path.exists(seg_path), seg_path

            seg = imageio.imread(seg_path)[z]

            bd, eva = _eval_seg(seg, eval_path)
            segmentations[seg_name] = seg
            boundaries[seg_name] = bd
            eval_im[seg_name] = eva

        v = napari.Viewer()
        v.add_image(image)
        for seg_name, bd in boundaries.items():
            v.add_labels(bd, name=seg_name, colormap={1: "cyan"})
            v.add_labels(eval_im[seg_name], name=f"{seg_name}_eval", colormap={1: "green", 2: "red"})
        v.title = Path(path).stem
        napari.run()


def main():
    # sgn_comparison()
    ihc_comparison()


if __name__ == "__main__":
    main()
