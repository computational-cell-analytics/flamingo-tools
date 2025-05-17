import os
from glob import glob

import torch
from torch_em.util import load_model
from flamingo_tools.training.domain_adaptation import mean_teacher_adaptation


def get_paths():
    root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops"
    folders = ["fHC", "iDISCO", "microwave-fHC", "microwave-iDISCO"]
    paths = []
    for folder in folders:
        paths.extend(glob(os.path.join(root, folder, "*.tif")))
    return paths[:-1], paths[-1:]


def run_training(name):
    patch_shape = (64, 128, 128)
    batch_size = 8
    source_checkpoint = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/cochlea_distance_unet_SGN_March2025Model"  # noqa

    train_paths, val_paths = get_paths()
    mean_teacher_adaptation(
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
    name = "sgn-adapted-model"
    if args.export_path is None:
        run_training(name)
    else:
        export_model(name, args.export_path)


if __name__ == "__main__":
    main()
