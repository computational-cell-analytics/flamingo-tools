import argparse
from datetime import datetime

import numpy as np
from micro_sam.training import default_sam_loader, train_sam
from train_distance_unet import get_image_and_label_paths, select_paths

ROOT_CLUSTER = "/scratch-grete/usr/nimcpape/data/moser/lightsheet/training"


def raw_transform(x):
    x = x.astype("float32")
    min_, max_ = np.percentile(x, 1), np.percentile(x, 99)
    x -= min_
    x /= max_
    x = np.clip(x, 0, 1)
    return x * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", "-i", help="The root folder with the annotated training crops.",
        default=ROOT_CLUSTER,
    )
    parser.add_argument(
        "--name", help="Optional name for the model to be trained. If not given the current date is used."
    )
    parser.add_argument(
        "--n_objects_per_batch", "-n", type=int, default=15,
        help="The number of objects to use during training. Set it to a lower value if you run out of GPU memory."
        "The default value is 15."
    )
    args = parser.parse_args()

    root = args.root
    run_name = datetime.now().strftime("%Y%m%d") if args.name is None else args.name
    name = f"cochlea_micro_sam_{run_name}"
    n_objects_per_batch = args.n_objects_per_batch

    image_paths, label_paths = get_image_and_label_paths(root)
    train_image_paths, train_label_paths = select_paths(image_paths, label_paths, split="train", filter_empty=True)
    val_image_paths, val_label_paths = select_paths(image_paths, label_paths, split="val", filter_empty=True)

    patch_shape = (1, 256, 256)
    max_sampling_attempts = 2500

    train_loader = default_sam_loader(
        raw_paths=train_image_paths, raw_key=None, label_paths=train_label_paths, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=True,
        raw_transform=raw_transform,
        num_workers=6, batch_size=1, is_train=True,
        max_sampling_attempts=max_sampling_attempts,
    )
    val_loader = default_sam_loader(
        raw_paths=val_image_paths, raw_key=None, label_paths=val_label_paths, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=True,
        raw_transform=raw_transform,
        num_workers=6, batch_size=1, is_train=False,
        max_sampling_attempts=max_sampling_attempts,
    )

    train_sam(
        name=name, model_type="vit_b_lm", train_loader=train_loader, val_loader=val_loader,
        n_epochs=50, n_objects_per_batch=n_objects_per_batch,
        save_root=".",
    )


if __name__ == "__main__":
    main()
