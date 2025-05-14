import os
import tempfile
from glob import glob

import tifffile
from elf.io import open_file
from flamingo_tools.segmentation import run_unet_prediction

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops"
MODEL_PATH = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/cochlea_distance_unet_SGN_March2025Model"  # noqa

SAVE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops/segmentations"


def check_data():
    files = glob(os.path.join(ROOT, "**/*.tif"), recursive=True)
    for ff in files:
        rel_path = sorted(os.path.relpath(ff, ROOT))
        shape = tifffile.memmap(ff).shape
        print(rel_path, shape)


def segment_crop(input_file):
    fname = os.path.relpath(input_file, ROOT)
    out_file = os.path.join(SAVE_ROOT, fname)
    if "segmentations" in input_file:
        return
    if os.path.exists(out_file):
        return

    print("Run prediction for", input_file)
    os.makedirs(os.path.split(out_file)[0], exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_folder:
        run_unet_prediction(
            input_file, input_key=None, output_folder=tmp_folder,
            model_path=MODEL_PATH, min_size=1000, use_mask=False,
        )
        seg_path = os.path.join(tmp_folder, "segmentation.zarr")
        with open_file(seg_path, mode="r") as f:
            seg = f["segmentation"][:]

    print("Writing output to", out_file)
    tifffile.imwrite(out_file, seg, bigtiff=True)


def segment_all():
    files = sorted(glob(os.path.join(ROOT, "**/*.tif"), recursive=True))
    for ff in files:
        segment_crop(ff)


def main():
    # check_data()
    segment_all()


if __name__ == "__main__":
    main()
