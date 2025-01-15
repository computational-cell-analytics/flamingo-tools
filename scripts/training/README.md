# 3D U-Net Training for Cochlea Data

This folder contains the scripts for training a 3D U-Net for cell segmentation in the cochlea data.
It contains two relevant scripts:
- `check_training_data.py`, which visualizes the training data and annotations in napari.
- `train_distance_unet.py`, which trains the 3D U-Net.
- `train_micro_sam.py`, which fine-tunes a micro-sam model on the data.

Both scripts accept the argument `-i /path/to/data`, to specify the root folder with the training data. For example, run `python train_distance_unet.py -i /path/to/data` for training. The scripts will consider all tif files in the sub-folders of the root folder for training.
They will load the **image data** according to the following rules:
- Files with the ending `_annotations.tif` or `_cp_masks.tif` will not be considered as image data.
- The other files will be considered as image data, if a corresponding file with ending `_annotations.tif` can be found. If it cannot be found the file will be excluded; the scripts will print the name of all files being excluded.

The training script will save the trained model in `checkpoints/cochlea_distance_unet_<CURRENT_DATE>`, e.g. `checkpoints/cochlea_distance_unet_20250115`.
For further options for the scripts run `python check_training_data.py -h` / `python train_distance_unet.py -h`.

The script `train_micro_sam.py` works similar to the U-Net training script. It saves the finetuned model for annotation with `micro_sam` to `checkpoints/`.
