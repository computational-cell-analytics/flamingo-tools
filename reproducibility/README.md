# Scripts to ensure/facilitate the reproducbility of pre-/post-processing

The scripts within these folders document steps involved in the pre-/post-processing or analysis of data.
The folders contain a script for reproducibility as well as dictionaries in JSON format, which contain script parameters and additional information.

## Extraction of blocks from a 3D volume

The extraction of blocks from a 3D volume is required for the creation of annotations, empty crops, or others regions of interest.

Usage:
```
python repro_block_extraction.py --input <JSON-file> --output <out-dir>
``` 

## Post-processing of SGN segmentation

The post-processing of the SGN segmentation may involve the erosion of the segmentation to exclude artifacts, the variation of the minimal number of nodes within a component, or the minimal distance between nodes to consider them the same component.

Usage:
 ```
python repro_postprocess_sgn_v1.py --input <JSON-file> --output <out-dir>
``` 

