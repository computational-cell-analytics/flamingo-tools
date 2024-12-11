# Flamingo Tools

Data processing for light-sheet microscopy, specifically for data from [Flamingo microscopes](https://huiskenlab.com/flamingo/).

The `flamingo_tools` library implements functionality for:
- converting the lightsheet data into a format compatible with [BigDataViewer](https://imagej.net/plugins/bdv/) and [BigStitcher](https://imagej.net/plugins/bigstitcher/).
- Cell / nucleus segmentation via a 3D U-net.
- ... and more functionality is planned!

This is work in progress!


## Requirements & Installation

You need a python environment with the following dependencies: [pybdv](https://github.com/constantinpape/pybdv) and [z5py](https://github.com/constantinpape/z5).
You install these dependencies with [mamba](https://github.com/mamba-org/mamba) or [conda](https://docs.conda.io/en/latest/) via: 
```bash
conda install -c conda-forge z5py pybdv
```
(for an existing conda environment). You can also set up a new environment with all required dependencies using the file `environment.yaml`:
```bash
conda env create -f environment.yaml
```
This will create the environment `flamingo`, which you can then activate via `conda activate flamingo`.
Finally, to install `flamingo_tools` into the environment run
```bash
pip install -e .
```

## Usage

We provide a command line tool, `convert_flamingo`, for converting data from the flamingo microscope to a data format compatible with BigDataViewer / BigStitcher:
```bash
convert_flamingo -i /path/to/data -o /path/to/output.n5 --file_ext .tif
```
Here, `/path/to/data` is the filepath to the folder with the flamingo data to be converted, `/path/to/output.n5` is the filepath where the converted data will be stored, and `--file_ext .tif` declares that the files are stored as tif stacks.
Use `--file_ext .raw` isntead if the data is stored in raw files.

The data will be converted to the [bdv.n5 format](https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md).
It can be opened with BigDataViewer via `Plugins->BigDataViewer->Open XML/HDF5`.
Or with BigStitcher as described [here](https://imagej.net/plugins/bigstitcher/open-existing).

You can also check out the following example scripts:
- `create_synthetic_data.py`: create small synthetic test data to check that the scripts work. 
- `convert_flamingo_data_examples.py`: convert flamingo data to a file format comatible with BigDataViewer / BigStitcher with parameters defined in the python script. Contains two example functions:
    - `convert_synthetic_data` to convert the synthetic data created via `create_synthetic_data.py`.
    - `convert_flamingo_data_moser` to convert sampled flamingo data from the Moser group.
- `load_data.py`: Example script for how to load sub-regions from the converted data into python.

For advanced examples to segment data with a U-Net, check out the `scripts` folder.
