import os
import sys

from detection_dataset import DetectionDataset

# sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")

from utils.training.training import supervised_training  # noqa

TRAIN_ROOT = "./training_data/images"
LABEL_ROOT = "./training_data/labels"


def get_paths(split):
    file_names = [
        "4.1L_apex_IHCribboncount_Z",
        "4.1L_base_IHCribbons_Z",
        "4.1L_mid_IHCribboncount_Z",
        "4.2R_apex_IHCribboncount_Z",
        "4.2R_apex_IHCribboncount_Z",
        "6.2R_apex_IHCribboncount_Z",
        "6.2R_base_IHCribbons_Z",
    ]
    image_paths = [os.path.join(TRAIN_ROOT, f"{fname}.zarr") for fname in file_names]
    label_paths = [os.path.join(LABEL_ROOT, f"{fname}.csv") for fname in file_names]

    if split == "train":
        image_paths = image_paths[:-1]
        label_paths = label_paths[:-1]
    else:
        image_paths = image_paths[-1:]
        label_paths = label_paths[-1:]

    return image_paths, label_paths


# TODO maybe add a sampler for the label data
def train():

    model_name = "synapse_detection_v1"

    train_paths, train_label_paths = get_paths("train")
    val_paths, val_label_paths = get_paths("val")
    # We need to give the paths for the test loader, although it's never used.
    test_paths, test_label_paths = val_paths, val_label_paths

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [40, 112, 112]
    batch_size = 32
    check = False

    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths=train_label_paths,
        val_paths=val_paths,
        val_label_paths=val_label_paths,
        raw_key="raw",
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        lr=1e-4,
        n_iterations=int(5e4),
        out_channels=1,
        augmentations=None,
        eps=1e-5,
        sigma=1,
        lower_bound=None,
        upper_bound=None,
        test_paths=test_paths,
        test_label_paths=test_label_paths,
        # save_root="",
        dataset_class=DetectionDataset,
        n_samples_train=3200,
        n_samples_val=160,
    )


def main():
    train()


if __name__ == "__main__":
    main()
