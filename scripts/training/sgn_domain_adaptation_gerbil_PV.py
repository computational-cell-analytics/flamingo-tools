import os
from glob import glob

import torch
from torch_em.util import load_model
from flamingo_tools.training import mean_teacher_training


def get_paths():
    root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/SGN"
    folders = ["2025-06-SGN_domain_gerbil_PV"]
    train_paths = []
    val_paths = []
    for folder in folders:
        train_paths.extend(glob(os.path.join(root, folder, "train", "*.tif")))
        val_paths.extend(glob(os.path.join(root, folder, "val", "*.tif")))
    return train_paths, val_paths


def run_training(name):
    patch_shape = (64, 128, 128)
    batch_size = 8
    source_checkpoint = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/v2_cochlea_distance_unet_SGN_supervised_2025-05-27"

    train_paths, val_paths = get_paths()
    mean_teacher_training(
        name=name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        patch_shape=patch_shape,
        source_checkpoint=source_checkpoint,
        batch_size=batch_size,
        n_iterations=int(2.5e4),
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
    name = "sgn-adapted-model_gerbil_PV"
    if args.export_path is None:
        run_training(name)
    else:
        export_model(name, args.export_path)


if __name__ == "__main__":
    main()
