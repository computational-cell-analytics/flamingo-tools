import os
from glob import glob

import torch_em

from torch_em.model import UNet3d

# DATA_ROOT = "/home/pape/Work/data/moser/lightsheet"
DATA_ROOT = "/scratch-grete/usr/nimcpape/data/moser/lightsheet"


def get_paths(image_paths, label_paths, split, filter_empty):
    if filter_empty:
        image_paths = [imp for imp in image_paths if "empty" not in imp]
        label_paths = [imp for imp in label_paths if "empty" not in imp]
    assert len(image_paths) == len(label_paths)

    n_files = len(image_paths)

    train_fraction = 0.8
    val_fraction = 0.1

    n_train = int(train_fraction * n_files)
    n_val = int(val_fraction * n_files)
    if split == "train":
        image_paths = image_paths[:n_train]
        label_paths = label_paths[:n_train]

    elif split == "val":
        image_paths = image_paths[n_train:(n_train + n_val)]
        label_paths = label_paths[n_train:(n_train + n_val)]

    return image_paths, label_paths


def get_loader(split, patch_shape, batch_size, filter_empty, train_on=["default"]):
    image_paths, label_paths = [], []

    if "default" in train_on:
        all_image_paths = sorted(glob(os.path.join(DATA_ROOT, "images", "*.tif")))
        all_label_paths = sorted(glob(os.path.join(DATA_ROOT, "masks", "*.tif")))
        this_image_paths, this_label_paths = get_paths(all_image_paths, all_label_paths, split, filter_empty)
        image_paths.extend(this_image_paths)
        label_paths.extend(this_label_paths)

    if "downsampled" in train_on:
        all_image_paths = sorted(glob(os.path.join(DATA_ROOT, "images_s2", "*.tif")))
        all_label_paths = sorted(glob(os.path.join(DATA_ROOT, "masks_s2", "*.tif")))
        this_image_paths, this_label_paths = get_paths(all_image_paths, all_label_paths, split, filter_empty)
        image_paths.extend(this_image_paths)
        label_paths.extend(this_label_paths)

    label_transform = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, foreground=True,
        )

    if split == "train":
        n_samples = 250 * batch_size
    elif split == "val":
        n_samples = 20 * batch_size

    sampler = torch_em.data.sampler.MinInstanceSampler(p_reject=0.95)
    loader = torch_em.default_segmentation_loader(
        raw_paths=image_paths, raw_key=None, label_paths=label_paths, label_key=None,
        batch_size=batch_size, patch_shape=patch_shape, label_transform=label_transform,
        n_samples=n_samples, num_workers=4, shuffle=True,
        sampler=sampler
    )
    return loader


def main(check_loaders=False):
    # Parameters for training:
    n_iterations = 1e5
    batch_size = 8
    filter_empty = False
    train_on = ["downsampled"]
    # train_on = ["downsampled", "default"]

    patch_shape = (32, 128, 128) if "downsampled" in train_on else (64, 128, 128)

    # The U-Net.
    model = UNet3d(in_channels=1, out_channels=3, initial_features=32, final_activation="Sigmoid")

    # Create the training loader with train and val set.
    train_loader = get_loader(
        "train", patch_shape, batch_size, filter_empty=filter_empty, train_on=train_on
    )
    val_loader = get_loader(
        "val", patch_shape, batch_size, filter_empty=filter_empty, train_on=train_on
    )

    if check_loaders:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, 3)
        check_loader(val_loader, 3)
        return

    loss = torch_em.loss.distance_based.DiceBasedDistanceLoss(mask_distances_in_bg=True)

    # Create the trainer.
    name = "cochlea_distance_unet"
    if filter_empty:
        name += "-filter-empty"
    if train_on == ["downsampled"]:
        name += "-train-downsampled"

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
    main(check_loaders=False)
