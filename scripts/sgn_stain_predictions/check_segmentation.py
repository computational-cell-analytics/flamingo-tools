import os
from glob import glob

import imageio.v3 as imageio
import napari


ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops"
SAVE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/LS_sampleprepcomparison_crops/segmentations"


def main():
    files = sorted(glob(os.path.join(ROOT, "**/*.tif")))
    for ff in files:
        if "segmentations" in ff:
            return
        print("Visualizing", ff)
        rel_path = os.path.relpath(ff, ROOT)
        seg_path = os.path.join(SAVE_ROOT, rel_path)

        image = imageio.imread(ff)
        if os.path.exists(seg_path):
            seg = imageio.imread(seg_path)
        else:
            seg = None

        v = napari.Viewer()
        v.add_image(image)
        if seg is not None:
            v.add_labels(seg)
        napari.run()


if __name__ == "__main__":
    main()
