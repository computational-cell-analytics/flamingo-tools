import os
from flamingo_tools.validation import fetch_data_for_evaluation, evaluate_annotated_slice

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1Validation"
TEST_ANNOTATION = os.path.join(ROOT, "AnnotationsEK/MAMD58L_PV_z771_base_full_annotationsEK.csv")


def main():
    segmentation, annotations = fetch_data_for_evaluation(TEST_ANNOTATION, cache_path="./seg.tif")
    # TODO visualize in napari locally.
    res = evaluate_annotated_slice(segmentation, annotations)
    print(res)


if __name__ == "__main__":
    main()
