import os
from glob import glob

import imageio.v3 as imageio
import napari


ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops"
SAVE_ROOT1 = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops/segmentations"  # noqa
SAVE_ROOT2 = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops/segmentations_v2"  # noqa


def main():
    files = sorted(glob(os.path.join(ROOT, "**/*.tif")))
    for ff in files:
        if "segmentations" in ff:
            return
        print("Visualizing", ff)
        rel_path = os.path.relpath(ff, ROOT)
        seg_path1 = os.path.join(SAVE_ROOT1, rel_path)
        seg_path2 = os.path.join(SAVE_ROOT2, rel_path)

        print("Load raw")
        image = imageio.imread(ff)
        print("Load segmentation 1")
        seg1 = imageio.imread(seg_path1) if os.path.exists(seg_path1) else None
        print("Load segmentation 2")
        seg2 = imageio.imread(seg_path2) if os.path.exists(seg_path2) else None

        v = napari.Viewer()
        v.add_image(image)
        if seg1 is not None:
            v.add_labels(seg1, name="original")
        if seg2 is not None:
            v.add_labels(seg2, name="adapted")
        napari.run()


if __name__ == "__main__":
    main()
