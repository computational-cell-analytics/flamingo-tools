# OME-Challenge

Scripts for converting flamingo data for the [OME-NGFF-Challenge](https://forum.image.sc/t/ome2024-ngff-challenge/97363):
- `convert_data.py`: to convert the data from the flamingo tif format to ome-zarr-v3 (via ome-zarr-v2 and the challenge converter tool)
- `create_metadata.py`: to add additional top-level metadata to keep track of the different tiles, timepoints etc.
- `upload_data.py`: to upload the data to s3 (needs credentials not stored in this repository)
