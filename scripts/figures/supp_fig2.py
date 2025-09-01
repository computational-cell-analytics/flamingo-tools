import json
import os

from glob import glob
from pathlib import Path

import numpy as np


# FIXME something is off with cellpose-sam runtimes
def runtimes_sgn():
    for_comparison = ["distance_unet", "micro-sam", "cellpose3", "cellpose-sam", "stardist"]

    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    val_sgn_dir = f"{cochlea_dir}/predictions/val_sgn"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation"

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))

    runtimes = {name: [] for name in for_comparison}

    for path in image_paths:
        eval_fname = Path(path).stem + "_dic.json"
        for seg_name in for_comparison:
            eval_path = os.path.join(val_sgn_dir, seg_name, eval_fname)
            with open(eval_path, "r") as f:
                result = json.load(f)
            rt = result["time"]
            runtimes[seg_name].append(rt)

    for name, rts in runtimes.items():
        print(name, ":", np.mean(rts), "+-", np.std(rts))


def runtimes_ihc():
    for_comparison = ["distance_unet_v3", "micro-sam", "cellpose3", "cellpose-sam"]

    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    val_sgn_dir = f"{cochlea_dir}/predictions/val_ihc"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationIHCs"

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))

    runtimes = {name: [] for name in for_comparison}

    for path in image_paths:
        eval_fname = Path(path).stem + "_dic.json"
        for seg_name in for_comparison:
            eval_path = os.path.join(val_sgn_dir, seg_name, eval_fname)
            if not os.path.exists(eval_path):
                continue
            with open(eval_path, "r") as f:
                result = json.load(f)
            rt = result["time"]
            runtimes[seg_name].append(rt)

    for name, rts in runtimes.items():
        print(name, ":", np.mean(rts), "+-", np.std(rts))


def main():
    print("SGNs:")
    runtimes_sgn()
    print()
    print("IHCs:")
    runtimes_ihc()


if __name__ == "__main__":
    main()
