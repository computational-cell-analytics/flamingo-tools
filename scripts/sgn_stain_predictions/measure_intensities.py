import os
from glob import glob

import tifffile
from flamingo_tools.measurements import compute_object_measures_impl


ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops"
SAVE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops/segmentations"


def measure_intensities(ff):
    rel_path = os.path.relpath(ff, ROOT)
    out_path = os.path.join("./measurements", rel_path.replace(".tif", ".xlsx"))
    if os.path.exists(out_path):
        return

    print("Computing measurements for", rel_path)
    seg_path = os.path.join(SAVE_ROOT, rel_path)

    image_data = tifffile.memmap(ff)
    seg_data = tifffile.memmap(seg_path)

    table = compute_object_measures_impl(image_data, seg_data, n_threads=8)

    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    table.to_excel(out_path, index=False)


def main():
    files = sorted(glob(os.path.join(ROOT, "**/*.tif")))
    for ff in files:
        if "segmentations" in ff:
            return
        measure_intensities(ff)


if __name__ == "__main__":
    main()
