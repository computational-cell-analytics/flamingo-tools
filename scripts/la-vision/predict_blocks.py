import argparse
import os
from glob import glob

import imageio.v3 as imageio
import numpy as np

from skimage.segmentation import watershed
from skimage.measure import label
from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo


def _get_files(sgn=True):
    input_root = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/more-annotations/LA_VISION_M04"  # noqa
    input_root2 = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/more-annotations/LA_VISION_M04_2"  # noqa
    input_root3 = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/more-annotations/LA_VISION_Mar05"  # noqa

    input_root_ihc = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/more-annotations/LA_VISION_Mar05-ihc"  # noqa

    if sgn:
        input_files = glob(os.path.join(input_root, "*.tif")) +\
            glob(os.path.join(input_root2, "*.tif")) +\
            glob(os.path.join(input_root3, "*.tif"))
    else:
        input_files = glob(os.path.join(input_root_ihc, "*.tif"))

    return input_files


def predict_blocks(model_path, name, sgn=True):
    output_folder = os.path.join("./predictions", name)
    os.makedirs(output_folder, exist_ok=True)

    input_blocks = _get_files(sgn)

    model = None
    for path in input_blocks:
        out_path = os.path.join(output_folder, os.path.basename(path))
        if os.path.exists(out_path):
            continue
        if model is None:
            model = load_model(model_path)
        data = imageio.imread(path)
        pred = predict_with_halo(data, model, gpu_ids=[0], block_shape=[64, 128, 128], halo=[8, 32, 32])
        imageio.imwrite(out_path, pred, compression="zlib")


def _segment_impl(pred, dist_threshold=0.5):
    fg, center_dist, boundary_dist = pred
    mask = fg > 0.5

    seeds = label(np.logical_and(center_dist < dist_threshold, boundary_dist < dist_threshold))
    seg = watershed(boundary_dist, mask=mask, markers=seeds)

    return seg


def check_segmentation(name, sgn):
    import napari

    input_files = _get_files(sgn)

    output_folder = os.path.join("./predictions", name)

    for path in input_files:
        image = imageio.imread(path)
        pred_path = os.path.join(output_folder, os.path.basename(path))
        pred = imageio.imread(pred_path)
        if sgn:
            seg = _segment_impl(pred)
        else:
            seg = label(pred[0] > 0.5)
        v = napari.Viewer()
        v.add_image(image)
        v.add_image(pred)
        v.add_labels(seg)
        v.title = os.path.basename(path)
        napari.run()


# Model path for original training on low-res SGNs:
# /mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/training/checkpoints/cochlea_distance_unet_low-res-sgn  # noqa
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", required=True)
    parser.add_argument("--model_path", "-m")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ihc", action="store_true")
    args = parser.parse_args()

    predict_blocks(args.model_path, args.name, sgn=not args.ihc)
    if args.check:
        check_segmentation(args.name, sgn=not args.ihc)


if __name__ == "__main__":
    main()
