import argparse
import os

import imageio.v3 as imageio
import napari

from elf.io import open_file


def _load_n5(path, key, lazy):
    data = open_file(path, "r")[key]
    if not lazy:
        data = data[:]
    return data


# will not work for very large images
def check_prediction(input_path, output_folder, check_downsampled=False, input_key=None, lazy=False):
    if input_key is None:
        input_ = imageio.imread(input_path)
    else:
        input_ = _load_n5(input_path, input_key, lazy)

    if check_downsampled:

        pred_path = os.path.join(output_folder, "predictions.zarr")
        prediction = _load_n5(pred_path, "prediction", lazy)

        seg_path = os.path.join(output_folder, "seg_downscaled.zarr")
        segmentation = _load_n5(seg_path, "segmentation", lazy)

        scale = (0.5, 0.5, 0.5)

    else:
        seg_path = os.path.join(output_folder, "segmentation.zarr")
        segmentation = _load_n5(seg_path, "segmentation", lazy)
        with open_file(seg_path, "r") as f:
            if "segmentation_postprocessed" in f:
                seg_pp = _load_n5(seg_path, "segmentation_postprocessed", lazy)
            else:
                seg_pp = None

        scale = None
        prediction = None

    v = napari.Viewer()
    v.add_image(input_, scale=scale)
    if prediction is not None:
        v.add_image(prediction)
    v.add_labels(segmentation)
    if seg_pp is not None:
        v.add_labels(seg_pp)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-k", "--input_key", default=None)
    parser.add_argument("-c", "--check_downsampled", action="store_true")
    parser.add_argument("--lazy", action="store_true")

    args = parser.parse_args()
    check_prediction(
        args.input, args.output_folder, check_downsampled=args.check_downsampled,
        input_key=args.input_key, lazy=args.lazy
    )


if __name__ == "__main__":
    main()
