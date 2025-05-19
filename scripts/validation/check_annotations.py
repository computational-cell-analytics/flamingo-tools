import os

import imageio.v3 as imageio
import napari
import pandas as pd

# ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1Validation"
ROOT = "annotation_data"
TEST_ANNOTATION = os.path.join(ROOT, "AnnotationsEK/MAMD58L_PV_z771_base_full_annotationsEK.csv")


def check_annotation(image_path, annotation_path):
    annotations = pd.read_csv(annotation_path)[["axis-0", "axis-1", "axis-2"]].values

    image = imageio.imread(image_path)
    v = napari.Viewer()
    v.add_image(image)
    v.add_points(annotations)
    napari.run()


def main():
    check_annotation(os.path.join(ROOT, "MAMD58L_PV_z771_base_full.tif"), TEST_ANNOTATION)


if __name__ == "__main__":
    main()
