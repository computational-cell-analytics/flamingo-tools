# Flamingo Tools

Data processing for light-sheet microscopy, specifically for data from [Flamingo microscopes](https://huiskenlab.com/flamingo/).

This is work in progress!


## Converting Data to N5 for BigDataViewer and BigStitcher

The script `convert_flamingo_data.py` can be used to convert the tif stacks for all channels of a region to a data format that is compatible with BigDataViewer and BigStitcher.
To run this script you will need a python environment with the following dependencies: [pybdv](https://github.com/constantinpape/pybdv) and [z5py](https://github.com/constantinpape/z5).
You can install these with [conda](https://docs.conda.io/en/latest/) / [mamba](https://github.com/mamba-org/mamba) via `mamba install -c conda-forge z5py pybdv`.
You can also set up a new environment with these dependencies using the file `environment.yaml`:
```bash
$ mamba env create -f environment.yaml
```
Once you have set up the environment you can run the script via `python convert_to_bdv_n5.py`. Note: the script contains the function `convert_region_to_bdv_n5`, which converts the tif stacks for the channels of a region/tile.
The two functions `convert_first_samples` and `convert_synthetic_data` show examples for how to use this function: the former was used to create the first converted volume that I have shared, the latter converts synthetic test data.
The synthetic test data can be created with the script `create_synthetic_data.py` by running `python create_synthetic_data.py`.

Once the data is converted it can be opened with BigDataViewer through Fiji via `Plugins->BigDataViewer->Open XML/HDF5`.
