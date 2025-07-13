import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from elf.io import open_file
from elf.parallel.local_maxima import find_local_maxima
from flamingo_tools.segmentation.unet_prediction import prediction_impl, run_unet_prediction

INPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v3/images"  # noqa
GT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v3/labels"
OUTPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/SynapseValidation"

sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")
sys.path.append("../../synapse_marker_detection")


def pred_synapse_impl(input_path, output_folder):
    model_path = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/synapse_marker_detection/checkpoints/synapse_detection_v3"  # noqa
    input_key = "raw"

    block_shape = (32, 128, 128)
    halo = (16, 64, 64)

    prediction_impl(
        input_path, input_key, output_folder, model_path,
        scale=None, block_shape=block_shape, halo=halo,
        apply_postprocessing=False, output_channels=1,
    )

    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    input_ = open_file(output_path, "r")[prediction_key]

    detections = find_local_maxima(
        input_, block_shape=block_shape, min_distance=2, threshold_abs=0.5, verbose=True, n_threads=4,
    )
    # Save the result in mobie compatible format.
    detections = np.concatenate(
        [np.arange(1, len(detections) + 1)[:, None], detections[:, ::-1]], axis=1
    )
    detections = pd.DataFrame(detections, columns=["spot_id", "x", "y", "z"])

    detection_path = os.path.join(output_folder, "synapse_detection.tsv")
    detections.to_csv(detection_path, index=False, sep="\t")


def predict_synapses():
    files = sorted(glob(os.path.join(INPUT_ROOT, "*.zarr")))
    for ff in files:
        output_folder = os.path.join(OUTPUT_ROOT, Path(ff).stem)
        if os.path.exists(os.path.join(output_folder, "predictions.zarr", "prediction")):
            print("Synapse prediction in", ff, "already done")
            continue
        else:
            print("Predicting synapses in", ff)
        pred_synapse_impl(ff, output_folder)


def pred_ihc_impl(input_path, output_folder):
    model_path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/v2_cochlea_distance_unet_IHC_supervised_2025-05-21"  # noqa

    run_unet_prediction(
        input_path, input_key="raw_ihc", output_folder=output_folder, model_path=model_path, min_size=1000,
        seg_class="ihc", center_distance_threshold=0.5, boundary_distance_threshold=0.5,
    )


def predict_ihcs():
    files = sorted(glob(os.path.join(INPUT_ROOT, "*.zarr")))
    for ff in files:
        output_folder = os.path.join(OUTPUT_ROOT, f"{Path(ff).stem}_ihc")
        if os.path.exists(os.path.join(output_folder, "predictions.zarr", "prediction")):
            print("IHC segmentation in", ff, "already done")
            continue
        else:
            print("Segmenting IHCs in", ff)
        pred_ihc_impl(ff, output_folder)


def _filter_synapse_impl(detections, ihc_file, output_path):
    from flamingo_tools.segmentation.synapse_detection import map_and_filter_detections

    with open_file(ihc_file, mode="r") as f:
        if "segmentation_filtered" in f:
            print("Using filtered segmentation!")
            segmentation = open_file(ihc_file)["segmentation_filtered"][:]
        else:
            segmentation = open_file(ihc_file)["segmentation"][:]

    max_distance = 5  # 5 micrometer
    filtered_detections = map_and_filter_detections(segmentation, detections, max_distance=max_distance)
    filtered_detections.to_csv(output_path, index=False, sep="\t")


def filter_synapses():
    input_files = sorted(glob(os.path.join(INPUT_ROOT, "*.zarr")))
    for ff in input_files:
        ihc = os.path.join(OUTPUT_ROOT, f"{Path(ff).stem}_ihc", "segmentation.zarr")
        output_folder = os.path.join(OUTPUT_ROOT, Path(ff).stem)
        synapses = os.path.join(output_folder, "synapse_detection.tsv")
        synapses = pd.read_csv(synapses, sep="\t")
        output_path = os.path.join(output_folder, "filtered_synapse_detection.tsv")
        _filter_synapse_impl(synapses, ihc, output_path)


def filter_gt():
    input_files = sorted(glob(os.path.join(INPUT_ROOT, "*.zarr")))
    gt_files = sorted(glob(os.path.join(GT_ROOT, "*.csv")))
    for ff, gt in zip(input_files, gt_files):
        ihc = os.path.join(OUTPUT_ROOT, f"{Path(ff).stem}_ihc", "segmentation.zarr")
        output_folder, fname = os.path.split(gt)
        output_path = os.path.join(output_folder, fname.replace(".csv", "_filtered.tsv"))

        gt = pd.read_csv(gt)
        gt = gt.rename(columns={"axis-0": "z", "axis-1": "y", "axis-2": "x"})
        gt.insert(0, "spot_id", np.arange(1, len(gt) + 1))

        _filter_synapse_impl(gt, ihc, output_path)


def _check_prediction(input_file, ihc_file, detection_file):
    import napari

    synapses = pd.read_csv(detection_file, sep="\t")[["z", "y", "x"]].values

    vglut = open_file(input_file)["raw_ihc"][:]
    ctbp2 = open_file(input_file)["raw"][:]
    ihcs = open_file(ihc_file)["segmentation"][:]

    v = napari.Viewer()
    v.add_image(vglut)
    v.add_image(ctbp2)
    v.add_labels(ihcs)
    v.add_points(synapses)
    napari.run()


def check_predictions():
    input_files = sorted(glob(os.path.join(INPUT_ROOT, "*.zarr")))
    for ff in input_files:
        ihc = os.path.join(OUTPUT_ROOT, f"{Path(ff).stem}_ihc", "segmentation.zarr")
        synapses = os.path.join(OUTPUT_ROOT, Path(ff).stem, "filtered_synapse_detection.tsv")
        _check_prediction(ff, ihc, synapses)


def process_everything():
    predict_synapses()
    predict_ihcs()
    filter_synapses()
    filter_gt()


def main():
    # process_everything()
    check_predictions()


if __name__ == "__main__":
    main()
