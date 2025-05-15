import os

import imageio.v3 as imageio
import napari

from flamingo_tools.validation import fetch_data_for_evaluation, compute_matches_for_annotated_slice, for_visualization

# ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1Validation"
ROOT = "annotation_data"
TEST_ANNOTATION = os.path.join(ROOT, "AnnotationsEK/MAMD58L_PV_z771_base_full_annotationsEK.csv")


def main():
    image = imageio.imread(os.path.join(ROOT, "MAMD58L_PV_z771_base_full.tif"))
    segmentation, annotations = fetch_data_for_evaluation(TEST_ANNOTATION, cache_path="./seg.tif")

    # v = napari.Viewer()
    # v.add_image(image)
    # v.add_labels(segmentation)
    # v.add_points(annotations)
    # napari.run()

    matches = compute_matches_for_annotated_slice(segmentation, annotations)
    tps, fns = matches["tp_annotations"], matches["fn"]
    vis_segmentation, vis_points, seg_props, point_props = for_visualization(segmentation, annotations, matches)

    print("True positive annotations:")
    print(tps)
    print("False negative annotations:")
    print(fns)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(vis_segmentation, **seg_props)
    v.add_points(vis_points, **point_props)
    v.add_labels(segmentation, visible=False)
    v.add_points(annotations, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
