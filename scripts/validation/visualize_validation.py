import argparse
import os

import imageio.v3 as imageio
import napari

from flamingo_tools.validation import (
    fetch_data_for_evaluation, compute_matches_for_annotated_slice, for_visualization, parse_annotation_path
)

# ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1Validation"
ROOT = "annotation_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--cache_folder")
    args = parser.parse_args()
    cache_folder = args.cache_folder

    cochlea, slice_id = parse_annotation_path(args.annotation)
    cache_path = None if cache_folder is None else os.path.join(cache_folder, f"{cochlea}_{slice_id}.tif")

    image = imageio.imread(args.image)
    segmentation, annotations = fetch_data_for_evaluation(
        args.annotation, cache_path=cache_path, components_for_postprocessing=[1],
    )

    matches = compute_matches_for_annotated_slice(segmentation, annotations, matching_tolerance=5)
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
