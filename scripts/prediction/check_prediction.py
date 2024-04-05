import argparse
import os

import imageio.v3 as imageio
import napari

from elf.io import open_file


# will not work for very large images
def check_prediction(input_path, output_folder, input_key=None):
    if input_key is None:
        input_ = imageio.imread(input_path)
    else:
        input_ = open_file(input_path, "r")[input_key][:]

    # TODO only load if present
    pred_path = os.path.join(output_folder, "predictions.zarr")
    with open_file(pred_path, "r") as f:
        prediction = f["prediction"][:]

    # TODO only load if present
    seg_path = os.path.join(output_folder, "segmentation.zarr")
    with open_file(seg_path, "r") as f:
        segmentation = f["segmentation"][:]

    v = napari.Viewer()
    v.add_image(input_)
    v.add_image(prediction)
    v.add_labels(segmentation)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-k", "--input_key", default=None)

    args = parser.parse_args()
    check_prediction(args.input, args.output_folder, args.input_key)


if __name__ == "__main__":
    main()
