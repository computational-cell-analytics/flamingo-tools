import os
from glob import glob

import torch
from torch_em.util import load_model
from flamingo_tools.training import mean_teacher_training


def get_paths():
    root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/IHC/2025-05-IHC_semi-supervised"
    annotated_folders = ["annotated_train", "empty"]
    train_image = []
    train_label = []
    for folder in annotated_folders:
        with os.scandir(os.path.join(root, folder)) as direc:
            for entry in direc:
                if "annotations" not in entry.name and entry.is_file():
                    basename = os.path.basename(entry.name)
                    name_no_extension = ".".join(basename.split(".")[:-1])
                    label_name = name_no_extension + "_annotations.tif"
                    train_image.extend(glob(os.path.join(root, folder, entry.name)))
                    train_label.extend(glob(os.path.join(root, folder, label_name)))

    annotated_folders = ["annotated_val"]
    val_image = []
    val_label = []
    for folder in annotated_folders:
        with os.scandir(os.path.join(root, folder)) as direc:
            for entry in direc:
                if "annotations" not in entry.name and entry.is_file():
                    basename = os.path.basename(entry.name)
                    name_no_extension = ".".join(basename.split(".")[:-1])
                    label_name = name_no_extension + "_annotations.tif"
                    val_image.extend(glob(os.path.join(root, folder, entry.name)))
                    val_label.extend(glob(os.path.join(root, folder, label_name)))

    domain_folders = ["domain_Aleyna", "domain_Lennart"]
    paths_domain = []
    for folder in domain_folders:
        paths_domain.extend(glob(os.path.join(root, folder, "*.tif")))

    return train_image, train_label, val_image, val_label, paths_domain[:-2], paths_domain[-2:]


def run_training(name):
    patch_shape = (64, 128, 128)
    batch_size = 8

    super_train_img, super_train_label, super_val_img, super_val_label, unsuper_train, unsuper_val = get_paths()

    mean_teacher_training(
        name=name,
        unsupervised_train_paths=unsuper_train,
        unsupervised_val_paths=unsuper_val,
        patch_shape=patch_shape,
        supervised_train_image_paths=super_train_img,
        supervised_val_image_paths=super_val_img,
        supervised_train_label_paths=super_train_label,
        supervised_val_label_paths=super_val_label,
        batch_size=batch_size,
        n_iterations=int(1e5),
        n_samples_train=1000,
        n_samples_val=80,
    )


def export_model(name, export_path):
    model = load_model(os.path.join("checkpoints", name), state_key="teacher")
    torch.save(model, export_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--export_path")
    args = parser.parse_args()
    name = "IHC_semi-supervised_2025-05-22"
    if args.export_path is None:
        run_training(name)
    else:
        export_model(name, args.export_path)


if __name__ == "__main__":
    main()
