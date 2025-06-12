import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from elf.io import open_file
from elf.parallel.local_maxima import find_local_maxima
from flamingo_tools.segmentation.unet_prediction import prediction_impl, run_unet_prediction

INPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v2/images"  # noqa
OUTPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/SynapseValidation"

sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")
sys.path.append("../../synapse_marker_detection")


def pred_synapse_impl(input_path, output_folder):
    model_path = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/synapse_marker_detection/checkpoints/synapse_detection_v2"  # noqa
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
    files = glob(os.path.join(INPUT_ROOT, "*.zarr"))
    for ff in files:
        print("Segmenting", ff)
        output_folder = os.path.join(OUTPUT_ROOT, Path(ff).stem)
        pred_synapse_impl(ff, output_folder)


def pred_ihc_impl(input_path, output_folder):
    model_path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/v2_cochlea_distance_unet_IHC_supervised_2025-05-21"  # noqa

    run_unet_prediction(
        input_path, input_key=None, output_folder=output_folder, model_path=model_path, min_size=1000,
        seg_class="ihc", center_distance_threshold=0.5, boundary_distance_threshold=0.5,
    )


def predict_ihcs():
    files = [
        "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/ImageCropsIHC_synapses/M226R_IHC-synapsecrops/M226R_base_p800_Vglut3.tif",  # noqa
        "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/ImageCropsIHC_synapses/M226R_IHC-synapsecrops/M226R_apex_p1268_Vglut3.tif",  # noqa
    ]
    for ff in files:
        print("Segmenting", ff)
        output_folder = os.path.join(OUTPUT_ROOT, Path(ff).stem)
        pred_ihc_impl(ff, output_folder)


# TODO also filter GT
def filter_synapses():
    pass


def check_predictions():
    pass


def main():
    # predict_synapses()
    predict_ihcs()


if __name__ == "__main__":
    main()
