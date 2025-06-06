import argparse
import os
from glob import glob

import napari
import tifffile

from flamingo_tools.validation import (
    fetch_data_for_evaluation, compute_matches_for_annotated_slice, for_visualization, parse_annotation_path
)

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"


def _match_image_path(annotation_path):
    all_files = glob(os.path.join(ROOT, "*.tif"))
    prefix = os.path.basename(annotation_path).split("_")[:-3]
    prefix = "_".join(prefix)
    matches = [path for path in all_files if os.path.basename(path).startswith(prefix)]
    # if len(matches) != 1:
    #     breakpoint()
    assert len(matches) == 1, f"{prefix}: {len(matches)}"
    return matches[0]


def visualize_anotation(annotation_path, cache_folder):
    print("Checking", annotation_path)
    cochlea, slice_id = parse_annotation_path(annotation_path)
    cache_path = None if cache_folder is None else os.path.join(cache_folder, f"{cochlea}_{slice_id}.tif")

    image_path = _match_image_path(annotation_path)

    segmentation, annotations = fetch_data_for_evaluation(
        annotation_path, cache_path=cache_path, components_for_postprocessing=[1],
    )

    image = tifffile.memmap(image_path)
    if segmentation.ndim == 2:
        image = image[image.shape[0] // 2]
    assert image.shape == segmentation.shape, f"{image.shape}, {segmentation.shape}"

    matches = compute_matches_for_annotated_slice(segmentation, annotations, matching_tolerance=5)
    vis_segmentation, vis_points, seg_props, point_props = for_visualization(segmentation, annotations, matches)

    # tps, fns = matches["tp_annotations"], matches["fn"]
    # print("True positive annotations:")
    # print(tps)
    # print("False negative annotations:")
    # print(fns)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(vis_segmentation, **seg_props)
    v.add_points(vis_points, **point_props)
    v.add_labels(segmentation, visible=False)
    v.add_points(annotations, visible=False)
    v.title = os.path.relpath(annotation_path, ROOT)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", nargs="+")
    parser.add_argument("--cache_folder")
    args = parser.parse_args()
    cache_folder = args.cache_folder

    if args.annotations is None:
        annotation_paths = sorted(glob(os.path.join(ROOT, "**", "*.csv"), recursive=True))
    else:
        annotation_paths = args.annotations

    for annotation_path in annotation_paths:
        visualize_anotation(annotation_path, cache_folder)


if __name__ == "__main__":
    main()
