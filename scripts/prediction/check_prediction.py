import argparse
import os

import imageio.v3 as imageio
import napari

from elf.io import open_file


# will not work for very large images
def check_prediction(input_path, output_folder, check_downsampled=False, input_key=None):
    if input_key is None:
        input_ = imageio.imread(input_path)
    else:
        input_ = open_file(input_path, "r")[input_key][:]

    if check_downsampled:

        pred_path = os.path.join(output_folder, "predictions.zarr")
        with open_file(pred_path, "r") as f:
            prediction = f["prediction"][:]

        seg_path = os.path.join(output_folder, "seg_downscaled.zarr")
        with open_file(seg_path, "r") as f:
            segmentation = f["segmentation"][:]

        scale = (0.5, 0.5, 0.5)

    else:
        seg_path = os.path.join(output_folder, "segmentation.zarr")
        with open_file(seg_path, "r") as f:
            segmentation = f["segmentation"][:]
        scale = None
        prediction = None

    v = napari.Viewer()
    v.add_image(input_, scale=scale)
    if prediction is not None:
        v.add_image(prediction)
    v.add_labels(segmentation)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-k", "--input_key", default=None)
    parser.add_argument("-c", "--check_downsampled", action="store_true")

    args = parser.parse_args()
    check_prediction(args.input, args.output_folder, check_downsampled=args.check_downsampled, input_key=args.input_key)


if __name__ == "__main__":
    main()
