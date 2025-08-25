import argparse
import os
from glob import glob

import imageio.v3 as imageio
from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo

INPUT_ROOT = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/more-annotations/LA_VISION_M04"  # noqa


def predict_blocks(model_path, name):
    output_folder = os.path.join("./predictions", name)
    os.makedirs(output_folder, exist_ok=True)

    input_blocks = glob(os.path.join(INPUT_ROOT, "*.tif"))

    model = load_model(model_path)
    for path in input_blocks:
        data = imageio.imread(path)
        pred = predict_with_halo(data, model, gpu_ids=[0], block_shape=[64, 128, 128], halo=[8, 32, 32])
        out_path = os.path.join(output_folder, os.path.basename(path))
        imageio.imwrite(out_path, pred, compression="zlib")


def _segment_impl(pred, dist_threshold=0.5):
    import numpy as np
    from skimage.segmentation import watershed
    from skimage.measure import label

    fg, center_dist, boundary_dist = pred
    mask = fg > 0.5

    seeds = label(np.logical_and(center_dist < dist_threshold, boundary_dist < dist_threshold))
    seg = watershed(boundary_dist, mask=mask, markers=seeds)

    return seg


def check_segmentation(name):
    import napari

    input_blocks = sorted(glob(os.path.join(INPUT_ROOT, "*.tif")))

    output_folder = os.path.join("./predictions", name)
    pred = sorted(glob(os.path.join(output_folder, "*.tif")))

    for path, pred_path in zip(input_blocks, pred):
        image = imageio.imread(path)
        pred = imageio.imread(pred_path)
        seg = _segment_impl(pred)
        v = napari.Viewer()
        v.add_image(image)
        v.add_image(pred)
        v.add_labels(seg)
        napari.run()


# Model path for original training on low-res SGNs:
# /mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/training/checkpoints/cochlea_distance_unet_low-res-sgn  # noqa
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--name", "-n", required=True)
    args = parser.parse_args()

    # predict_blocks(args.model_path, args.name)
    check_segmentation(args.name)


if __name__ == "__main__":
    main()
