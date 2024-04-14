# Segmentation for large lightsheet volumes

## Training

Contains the scripts for training a U-Net that predicts foreground probabilties and normalized object distances.

## Prediction

Contains the scripts for running segmentation for a large volume with a distance prediction U-net. (Other scripts are work in progress.)

You can run it like this for input that is stored in n5/zarr:
```
python run_prediction_distance_unet.py -i /path/to/volume.n5 -k setup0/timepoint0/s0 -o /path/to/output_folder
```
Or for a tif file like this:
```
python run_prediction_distance_unet.py -i /path/to/volume.tif -o /path/to/output_folder
```
Here, `-i` specifies the input filepath, `-o` the folder where the results are saved and `-k` the internal path for a zarr or n5 file.
The result will be stored as `segmentation.zarr` in the output folder, where also the intermediate files `prediction.zarr` and `seeds.zarr` are saved.

In order to downsample the segmentation on the fly for the segmentation process you can use the argument `--scale`.
E.g. run the command 
```
python run_prediction_distance_unet.py ... --scale 2
```
to downsample the input by a factor of 2. Note that the segmentation result will be automatically rescaled back to the full shape of the input at the end.

You can use the script `to_tif.py` to convert the zarr object to a tif volume for easier viewing (won't work for very large volumes!).

## Installation

Needs [torch-em](https://github.com/constantinpape/torch-em) in the python environment. See [here](https://github.com/constantinpape/torch-em?tab=readme-ov-file#installation) for installation instructions. (If possible use `mamba` instead of `conda`.)
