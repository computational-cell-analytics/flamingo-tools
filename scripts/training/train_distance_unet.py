import argparse
import os
from datetime import datetime
from glob import glob

import torch_em
from flamingo_tools.training import get_supervised_loader, get_3d_model

ROOT_CLUSTER = "/scratch-grete/usr/nimcpape/data/moser/lightsheet/training"


def get_image_and_label_paths(root):
    exclude_names = ["annotations", "cp_masks"]
    all_image_paths = sorted(glob(os.path.join(root, "**/**.tif"), recursive=True))
    all_image_paths = [
        path for path in all_image_paths if not any(exclude in path for exclude in exclude_names)
    ]

    image_paths, label_paths = [], []
    label_extensions = ["_annotations.tif"]
    for path in all_image_paths:
        folder, fname = os.path.split(path)
        fname = os.path.splitext(fname)[0]
        label_path = None
        for ext in label_extensions:
            candidate_label_path = os.path.join(folder, f"{fname}{ext}")
            if os.path.exists(candidate_label_path):
                label_path = candidate_label_path
                break

        if label_path is None:
            print("Did not find annotations for", path)
            print("This image will not be used for training.")
        else:
            image_paths.append(path)
            label_paths.append(label_path)

    assert len(image_paths) == len(label_paths)
    return image_paths, label_paths


def get_image_and_label_paths_sep_folders(root):
    image_paths = sorted(glob(os.path.join(root, "images", "**", "*.tif"), recursive=True))
    label_paths = sorted(glob(os.path.join(root, "labels", "**", "*.tif"), recursive=True))
    assert len(image_paths) == len(label_paths)

    # import imageio.v3 as imageio
    # for imp, lp in zip(image_paths, label_paths):
    #     s1 = imageio.imread(imp).shape
    #     s2 = imageio.imread(lp).shape
    #     if s1 != s2:
    #         breakpoint()

    return image_paths, label_paths


def select_paths(image_paths, label_paths, split, filter_empty):
    if filter_empty:
        image_paths = [imp for imp in image_paths if "empty" not in imp]
        label_paths = [imp for imp in label_paths if "empty" not in imp]
    assert len(image_paths) == len(label_paths)

    n_files = len(image_paths)
    train_fraction = 0.85

    n_train = int(train_fraction * n_files)
    if split == "train":
        image_paths = image_paths[:n_train]
        label_paths = label_paths[:n_train]

    elif split == "val":
        image_paths = image_paths[n_train:]
        label_paths = label_paths[n_train:]

    return image_paths, label_paths


def get_loader(root, split, patch_shape, batch_size, filter_empty, separate_folders):
    if separate_folders:
        image_paths, label_paths = get_image_and_label_paths_sep_folders(root)
    else:
        image_paths, label_paths = get_image_and_label_paths(root)
    this_image_paths, this_label_paths = select_paths(image_paths, label_paths, split, filter_empty)

    assert len(this_image_paths) == len(this_label_paths)
    assert len(this_image_paths) > 0

    if split == "train":
        n_samples = 250 * batch_size
    elif split == "val":
        n_samples = 16 * batch_size

    return get_supervised_loader(this_image_paths, this_label_paths, patch_shape, batch_size, n_samples=n_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", "-i", help="The root folder with the annotated training crops.",
        default=ROOT_CLUSTER,
    )
    parser.add_argument(
        "--batch_size", "-b", help="The batch size for training. Set to 8 by default."
        "You may need to choose a smaller batch size to train on yoru GPU.",
        default=8, type=int,
    )
    parser.add_argument(
        "--check_loaders", "-l", action="store_true",
        help="Visualize the data loader output instead of starting a training run."
    )
    parser.add_argument(
        "--filter_empty", "-f", action="store_true",
        help="Whether to exclude blocks with empty annotations from the training process."
    )
    parser.add_argument(
        "--name", help="Optional name for the model to be trained. If not given the current date is used."
    )
    parser.add_argument("--separate_folders", action="store_true")
    args = parser.parse_args()
    root = args.root
    batch_size = args.batch_size
    check_loaders = args.check_loaders
    filter_empty = args.filter_empty
    run_name = datetime.now().strftime("%Y%m%d") if args.name is None else args.name

    # Parameters for training on A100.
    n_iterations = int(1e5)
    patch_shape = (48, 128, 128)

    # The U-Net.
    model = get_3d_model()

    # Create the training loader with train and val set.
    train_loader = get_loader(
        root, "train", patch_shape, batch_size, filter_empty=filter_empty, separate_folders=args.separate_folders
    )
    val_loader = get_loader(
        root, "val", patch_shape, batch_size, filter_empty=filter_empty, separate_folders=args.separate_folders
    )

    if check_loaders:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, 3)
        check_loader(val_loader, 3)
        return

    loss = torch_em.loss.distance_based.DiceBasedDistanceLoss(mask_distances_in_bg=True)

    # Create the trainer.
    name = f"cochlea_distance_unet_{run_name}"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
    )

    # Run the training.
    trainer.fit(iterations=n_iterations)


if __name__ == "__main__":
    main()
