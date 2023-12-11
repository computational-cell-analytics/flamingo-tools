# Flamingo Tools

Data processing for light-sheet microscopy, specifically for data from [Flamingo microscopes](https://huiskenlab.com/flamingo/).

The `flamingo_tools` library implements functionality for:
- converting the lightsheet data into a format compatible with [BigDataViewer](https://imagej.net/plugins/bdv/) and [BigStitcher](https://imagej.net/plugins/bigstitcher/).
- ... and more functionality is planned!

This is work in progress!


## Requirements & Installation

You need a python environment with the following dependencies: [pybdv](https://github.com/constantinpape/pybdv) and [z5py](https://github.com/constantinpape/z5).
You can for example install these dependencies with [mamba](https://github.com/mamba-org/mamba) (a faster implementation of [conda](https://docs.conda.io/en/latest/)) via: 
```bash
$ mamba install -c conda-forge z5py pybdv
```
You can also set up a new environment with these dependencies using the file `environment.yaml`:
```bash
$ mamba env create -f environment.yaml
```

## Usage

We provide the follwoing scripts:
- `create_synthetic_data.py`: create small synthetic test data to check that the scripts work. 
- `convert_flamingo_data.py`: convert flamingo data to a file format comatible with BigDataViewer / BigStitcher via command line interface. Run `python convert_flamingo_data.py -h` for details. 
- `convert_flamingo_data_examples.py`: convert flamingo data to a file format comatible with BigDataViewer / BigStitcher with parameters defined in the python script. Contains two example functions:
    - `convert_synthetic_data` to convert the synthetic data created via `create_synthetic_data.py`.
    - `convert_flamingo_data_moser` to convert sampled flamingo data from the Moser group.
- `load_data.py`: Example script for how to load sub-regions from the converted data into python.

The data will be converted to the [bdv.n5 format](https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md).
It can be opened with BigDataViewer via `Plugins->BigDataViewer->Open XML/HDF5`.
Or with BigStitcher as described [here](https://imagej.net/plugins/bigstitcher/open-existing).
