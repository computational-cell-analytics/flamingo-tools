import argparse
import os
from shutil import rmtree

import pybdv.metadata as bdv_metadata
import torch
import z5py

from flamingo_tools.segmentation import run_unet_prediction, filter_isolated_objects
from flamingo_tools.mobie import add_raw_to_mobie, add_segmentation_to_mobie

MOBIE_ROOT = "/mnt/lustre-emmy-hdd/projects/nim00007/data/moser/lightsheet/mobie"


def postprocess_seg(output_folder):
    print("Run segmentation postprocessing ...")
    seg_path = os.path.join(output_folder, "segmentation.zarr")
    seg_key = "segmentation"

    with z5py.File(seg_path, "r") as f:
        segmentation = f[seg_key][:]

    seg_filtered, n_pre, n_post = filter_isolated_objects(segmentation)

    with z5py.File(seg_path, "a") as f:
        chunks = f[seg_key].chunks
        f.create_dataset(
            "segmentation_postprocessed", data=seg_filtered, compression="gzip",
            chunks=chunks, dtype=seg_filtered.dtype
        )


def export_to_mobie(xml_path, segmentation_folder, scale, mobie_dataset, chunks):
    # Add to mobie:

    # - raw data (if not yet present)
    add_raw_to_mobie(
        mobie_project=MOBIE_ROOT,
        mobie_dataset=mobie_dataset,
        source_name="pv-channel",
        xml_path=xml_path,
        setup_id=0,
    )

    # TODO enable passing extra channel names
    # - additional channels
    setup_ids = bdv_metadata.get_setup_ids(xml_path)
    if len(setup_ids) > 1:
        extra_channel_names = ["gfp_channel", "myo_channel"]
        for i, setup_id in enumerate(setup_ids[1:]):
            add_raw_to_mobie(
                mobie_project=MOBIE_ROOT,
                mobie_dataset=mobie_dataset,
                source_name=extra_channel_names[i],
                xml_path=xml_path,
                setup_id=setup_id
            )

    # - segmentation and post-processed segmentation
    seg_path = os.path.join(segmentation_folder, "segmentation.zarr")
    seg_resolution = bdv_metadata.get_resolution(xml_path, setup_id=0)
    if scale == 1:
        seg_resolution = [2 * res for res in seg_resolution]
    unit = bdv_metadata.get_unit(xml_path, setup_id=0)

    seg_key = "segmentation"
    seg_name = "nuclei_fullscale" if scale == 0 else "nuclei_downscaled"
    add_segmentation_to_mobie(
        mobie_project=MOBIE_ROOT,
        mobie_dataset=mobie_dataset,
        source_name=seg_name,
        segmentation_path=seg_path,
        segmentation_key=seg_key,
        resolution=seg_resolution,
        unit=unit,
        scale_factors=4*[[2, 2, 2]],
        chunks=chunks,
    )

    seg_key = "segmentation_postprocessed"
    seg_name += "_postprocessed"
    add_segmentation_to_mobie(
        mobie_project=MOBIE_ROOT,
        mobie_dataset=mobie_dataset,
        source_name=seg_name,
        segmentation_path=seg_path,
        segmentation_key=seg_key,
        resolution=seg_resolution,
        unit=unit,
        scale_factors=4*[[2, 2, 2]],
        chunks=chunks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-s", "--scale", required=True, type=int)
    parser.add_argument("-m", "--mobie_dataset", required=True)
    parser.add_argument("--model")

    args = parser.parse_args()

    scale = args.scale
    if scale == 0:
        min_size = 1000
    elif scale == 1:
        min_size = 250
    else:
        raise ValueError

    xml_path = args.input
    assert os.path.splitext(xml_path)[1] == ".xml"
    input_path = bdv_metadata.get_data_path(xml_path, return_absolute_path=True)

    # TODO need to make sure that PV is always setup 0
    input_key = f"setup0/timepoint0/s{scale}"

    have_cuda = torch.cuda.is_available()
    chunks = z5py.File(input_path, "r")[input_key].chunks
    block_shape = tuple([2 * ch for ch in chunks]) if have_cuda else tuple(chunks)
    halo = (16, 64, 64) if have_cuda else (8, 32, 32)

    if args.model is not None:
        model = args.model
    else:
        if scale == 0:
            model = "../training/checkpoints/cochlea_distance_unet"
        else:
            model = "../training/checkpoints/cochlea_distance_unet-train-downsampled"

    run_unet_prediction(
        input_path, input_key, args.output_folder, model,
        scale=None, min_size=min_size,
        block_shape=block_shape, halo=halo,
    )

    postprocess_seg(args.output_folder)

    export_to_mobie(xml_path, args.output_folder, scale, args.mobie_dataset, chunks)

    # clean up: remove segmentation folders
    print("Cleaning up intermediate segmentation results")
    print("This may take a while, but everything else is done.")
    print("You can check the results in the MoBIE project already at:")
    print(f"{MOBIE_ROOT}:{args.mobie_dataset}")
    rmtree(args.output_folder)


if __name__ == "__main__":
    main()
