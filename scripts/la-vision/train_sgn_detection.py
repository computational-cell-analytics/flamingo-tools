import os
import sys
import json
from glob import glob

from sklearn.model_selection import train_test_split

sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")
sys.path.append("../synapse_marker_detection")

from utils.training.training import supervised_training  # noqa
from detection_dataset import DetectionDataset, MinPointSampler  # noqa

ROOT = "./la-vision-sgn-new"  # noqa

TRAIN = os.path.join(ROOT, "images")
TRAIN_EMPTY = os.path.join(ROOT, "empty_images")

LABEL = os.path.join(ROOT, "centroids")
LABEL_EMPTY = os.path.join(ROOT, "empty_centroids")


def _get_paths(split, train_folder, label_folder, n=None):
    image_paths = sorted(glob(os.path.join(train_folder, "*.tif")))
    label_paths = sorted(glob(os.path.join(label_folder, "*.csv")))
    assert len(image_paths) == len(label_paths)
    if n is not None:
        image_paths, label_paths = image_paths[:n], label_paths[:n]

    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, label_paths, test_size=1, random_state=42
    )

    if split == "train":
        image_paths = train_images
        label_paths = train_labels
    else:
        image_paths = val_images
        label_paths = val_labels

    return image_paths, label_paths


def get_paths(split):
    image_paths, label_paths = _get_paths(split, TRAIN, LABEL)
    empty_image_paths, empty_label_paths = _get_paths(split, TRAIN_EMPTY, LABEL_EMPTY, n=4)
    return image_paths + empty_image_paths, label_paths + empty_label_paths


def train():

    model_name = "sgn-low-res-detection-v1"

    train_paths, train_label_paths = get_paths("train")
    val_paths, val_label_paths = get_paths("val")
    # We need to give the paths for the test loader, although it's never used.
    test_paths, test_label_paths = val_paths, val_label_paths

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [48, 256, 256]
    batch_size = 8
    check = False

    checkpoint_path = f"./checkpoints/{model_name}"
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, "splits.json"), "w") as f:
        json.dump(
            {
                "train": {"images": train_paths, "labels": train_label_paths},
                "val": {"images": val_paths, "labels": val_label_paths},
            },
            f, indent=2, sort_keys=True
        )

    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths=train_label_paths,
        val_paths=val_paths,
        val_label_paths=val_label_paths,
        raw_key=None,
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        lr=1e-4,
        n_iterations=int(1e5),
        out_channels=1,
        augmentations=None,
        eps=1e-5,
        sigma=4,
        lower_bound=None,
        upper_bound=None,
        test_paths=test_paths,
        test_label_paths=test_label_paths,
        # save_root="",
        dataset_class=DetectionDataset,
        n_samples_train=3200,
        n_samples_val=160,
        sampler=MinPointSampler(min_points=1, p_reject=0.5),
    )


def main():
    train()


if __name__ == "__main__":
    main()
