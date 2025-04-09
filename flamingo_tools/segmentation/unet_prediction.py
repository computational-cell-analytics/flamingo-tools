import multiprocessing as mp
import os
import sys
import warnings
from concurrent import futures

import imageio.v3 as imageio
import elf.parallel as parallel
import numpy as np
import nifty.tools as nt
import vigra
import torch
import z5py
import zarr
import json

from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper
from elf.wrapper.resized_volume import ResizedVolume
from elf.io import open_file
from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo
from tqdm import tqdm
from inspect import getsourcefile

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(getsourcefile(lambda:0)))), "scripts", "prediction"))
import upload_to_s3

"""
Prediction using distance U-Net.
Parallelization using multiple GPUs is currently only possible by calling functions directly.
Functions for the parallelization end with '_slurm' and divide the process into preprocessing, prediction, and segmentation.
"""

class SelectChannel(SimpleTransformationWrapper):
    def __init__(self, volume, channel):
        self.channel = channel
        super().__init__(volume, lambda x: x[self.channel], with_channels=True)

    @property
    def shape(self):
        return self._volume.shape[1:]

    @property
    def chunks(self):
        return self._volume.chunks[1:]

    @property
    def ndim(self):
        return self._volume.ndim - 1


def prediction_impl(input_path, input_key, output_folder, model_path, scale, block_shape, halo, prediction_instances=1, slurm_task_id=0, mean=None, std=None, s3=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if os.path.isdir(model_path):
            model = load_model(model_path)
        else:
            model = torch.load(model_path, weights_only=False)

    mask_path = os.path.join(output_folder, "mask.zarr")
    image_mask = z5py.File(mask_path, "r")["mask"]

    if input_key is None:
        input_ = imageio.imread(input_path)
    elif s3 is not None:
        with zarr.open(input_path, mode="r") as f:
            input_ = f[input_key]
    else:
        input_ = open_file(input_path, "r")[input_key]

    if scale is None or scale == 1:
        original_shape = None
    else:
        original_shape = input_.shape
        new_shape = tuple(
            int(round(sh / scale)) for sh in original_shape
        )
        print("The input is processed downsampled by a factor of scale", scale)
        print("Corresponding to shape", new_shape, "instead of", original_shape)
        input_ = ResizedVolume(input_, shape=new_shape, order=3)
        image_mask = ResizedVolume(image_mask, new_shape, order=0)

    have_cuda = torch.cuda.is_available()

    if block_shape is None:
        block_shape = (128, 128, 128) if have_cuda else input_.chunks
    if halo is None:
        halo = (16, 32, 32)
    if have_cuda:
        print("Predict with GPU")
        gpu_ids = [0]
    else:
        print("Predict with CPU")
        gpu_ids = ["cpu"]

    if mean is None or std is None:
        # Compute the global mean and standard deviation.
        n_threads = min(16, mp.cpu_count())
        mean, std = parallel.mean_and_std(
            input_, block_shape=block_shape, n_threads=n_threads, verbose=True,
            mask=image_mask
        )
    print("Mean and standard deviation computed for the full volume:")
    print(mean, std)

    # Preprocess with fixed mean and standard deviation.
    def preprocess(raw):
        raw = raw.astype("float32")
        raw -= mean
        raw /= std
        return raw

    # Smooth the distance prediction channel.
    def postprocess(x):
        x[1] = vigra.filters.gaussianSmoothing(x[1], sigma=2.0)
        return x

    shape = input_.shape
    ndim = len(shape)

    blocking = nt.blocking([0] * ndim, shape, block_shape)
    n_blocks = blocking.numberOfBlocks
    if prediction_instances != 1:
        iteration_ids = [x.tolist() for x in np.array_split(list(range(n_blocks)), prediction_instances)]
        slurm_iteration = iteration_ids[slurm_task_id]
    else:
        slurm_iteration = list(range(n_blocks))

    output_path = os.path.join(output_folder, "predictions.zarr")
    with open_file(output_path, "a") as f:
        output = f.require_dataset(
            "prediction",
            shape=(3,) + input_.shape,
            chunks=(1,) + block_shape,
            compression="gzip",
            dtype="float32",
        )

        predict_with_halo(
            input_, model,
            gpu_ids=gpu_ids, block_shape=block_shape, halo=halo,
            output=output, preprocess=preprocess, postprocess=postprocess,
            mask=image_mask,
            iter_list=slurm_iteration,
        )

    return original_shape


def find_mask(input_path, input_key, output_folder, s3=None):
    mask_path = os.path.join(output_folder, "mask.zarr")
    f = z5py.File(mask_path, "a")

    mask_key = "mask"
    if mask_key in f:
        return

    if input_key is None:
        raw = imageio.imread(input_path)
        chunks = (64, 64, 64)
    elif s3 is not None:
        with zarr.open(input_path, mode="r") as fin:
            raw = fin[input_key]
        chunks = raw.chunks
    else:
        fin = open_file(input_path, "r")
        raw = fin[input_key]
        chunks = raw.chunks

    block_shape = tuple(2 * ch for ch in chunks)
    blocking = nt.blocking([0, 0, 0], raw.shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    ds_mask = f.create_dataset(mask_key, shape=raw.shape, compression="gzip", dtype="uint8", chunks=block_shape)

    # TODO more sophisticated criterion?!
    def find_mask_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data = raw[bb]
        max_ = np.percentile(data, 95)
        if max_ > 200:
            ds_mask[bb] = 1

    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(find_mask_block, range(n_blocks)), total=n_blocks))


def segmentation_impl(input_path, output_folder, min_size, original_shape=None):
    input_ = open_file(input_path, "r")["prediction"]

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())

    # The center distances as input for computing the seeds.
    center_distances = SelectChannel(input_, 1)
    block_shape = center_distances.chunks

    # Compute the seeds based on smoothed center distances < 0.5.
    seed_path = os.path.join(output_folder, "seeds.zarr")
    seed_file = open_file(os.path.join(seed_path), "a")
    seeds = seed_file.require_dataset(
        "seeds", shape=center_distances.shape, chunks=block_shape, compression="gzip", dtype="uint64"
    )

    fg_threshold = 0.5
    mask = ThresholdWrapper(SelectChannel(input_, 0), threshold=fg_threshold)

    parallel.label(
        data=ThresholdWrapper(center_distances, threshold=0.4, operator=np.less),
        out=seeds, block_shape=block_shape, mask=mask, verbose=True, n_threads=n_threads
    )

    # Run the watershed.
    if original_shape is None:
        seg_path = os.path.join(output_folder, "segmentation.zarr")
    else:
        seg_path = os.path.join(output_folder, "seg_downscaled.zarr")

    seg_file = open_file(seg_path, "a")
    seg = seg_file.create_dataset(
        "segmentation", shape=seeds.shape, chunks=block_shape, compression="gzip", dtype="uint64"
    )

    hmap = SelectChannel(input_, 2)
    halo = (2, 8, 8)
    parallel.seeded_watershed(
        hmap, seeds, out=seg, block_shape=block_shape, halo=halo, mask=mask, verbose=True,
        n_threads=n_threads,
    )

    if min_size > 0:
        parallel.size_filter(
            seg, seg, min_size=min_size, block_shape=block_shape, mask=mask,
            verbose=True, n_threads=n_threads, relabel=True,
        )

    if original_shape is not None:
        out_path = os.path.join(output_folder, "segmentation.zarr")

        # This logic should be refactored.
        output_seg = ResizedVolume(seg, shape=original_shape, order=0)
        with open_file(out_path, "a") as f:
            out_seg_volume = f.create_dataset(
                "segmentation", shape=original_shape, compression="gzip", dtype="uint64", chunks=block_shape,
            )
            blocking = parallel.common.get_blocking(output_seg, block_shape, roi=None, n_threads=n_threads)

            def write_block(block_id):
                block = blocking.getBlock(block_id)
                bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
                out_seg_volume[bb] = output_seg[bb]

            with futures.ThreadPoolExecutor(n_threads) as tp:
                tp.map(write_block, range(blocking.numberOfBlocks))


def calc_mean_and_std(
    input_path, input_key, output_folder,
    s3=None,
    ):
    """
    Calculate mean and standard deviation of full volume.
    Parameters are saved in 'mean_std.json' within the output folder.
    """
    json_file = os.path.join(output_folder, "mean_std.json")
    mask_path = os.path.join(output_folder, "mask.zarr")
    image_mask = z5py.File(mask_path, "r")["mask"]

    if input_key is None:
        input_ = imageio.imread(input_path)
    elif s3 is not None:
        with zarr.open(input_path, mode="r") as f:
            input_ = f[input_key]
    else:
        input_ = open_file(input_path, "r")[input_key]

    # Compute the global mean and standard deviation.
    n_threads = min(16, mp.cpu_count())
    mean, std = parallel.mean_and_std(
        input_, block_shape=tuple([2* i for i in input_.chunks]), n_threads=n_threads, verbose=True,
        mask=image_mask
    )
    ddict = {"mean":mean, "std":std}
    with open(json_file, "w") as f:
        json.dump(ddict, f)


def run_unet_prediction(
    input_path, input_key,
    output_folder, model_path,
    min_size, scale=None,
    block_shape=None, halo=None,
):
    os.makedirs(output_folder, exist_ok=True)

    find_mask(input_path, input_key, output_folder)

    original_shape = prediction_impl(
        input_path, input_key, output_folder, model_path, scale, block_shape, halo
    )

    pmap_out = os.path.join(output_folder, "predictions.zarr")
    segmentation_impl(pmap_out, output_folder, min_size=min_size, original_shape=original_shape)

#---Workflow for parallel prediction using slurm---

def run_unet_prediction_preprocess_slurm(
        input_path, input_key, output_folder,
        s3=None, s3_bucket_name=None, s3_service_endpoint=None, s3_credentials=None,
):
    """
    Pre-processing for the parallel prediction with U-Net models.
    Masks are stored in mask.zarr in the output folder.
    The mean and standard deviation are precomputed for later usage during prediction
    and stored in a JSON file within the output folder as mean_std.json.
    """
    if s3 is not None:
        bucket_name, service_endpoint, credentials = upload_to_s3.check_s3_credentials(s3_bucket_name, s3_service_endpoint, s3_credentials)

        input_path, fs = upload_to_s3.get_s3_path(input_path, bucket_name=bucket_name, service_endpoint=service_endpoint, credential_file=credentials)

    if not os.path.isdir(os.path.join(output_folder, "mask.zarr")):
        find_mask(input_path, input_key, output_folder, s3=s3)

    calc_mean_and_std(input_path, input_key, output_folder, s3=s3)


def run_unet_prediction_slurm(
    input_path, input_key, output_folder, model_path,
    scale=None,
    block_shape=None, halo=None, prediction_instances=1,
    s3=None, s3_bucket_name=None, s3_service_endpoint=None, s3_credentials=None,
):
    """
    Run prediction of distance U-Net for data stored locally or on an S3 bucket.

    :param str input_path: File path to input data
    :param str input_key: Input key for data in ome.zarr format
    :param str output_folder: Output folder for prediction.zarr
    :param str model_path: File path to distance U-Net model
    :param float scale:
    :param tuple block_shape:
    :param tuple halo:
    :param int prediction_instances: Number of workers for parallel computation within slurm array
    :param bool s3: Flag for accessing data on S3 bucket
    :param str s3_bucket_name: S3 bucket name. Optional if BUCKET_NAME has been exported
    :param str s3_service_endpoint: S3 service endpoint. Optional if SERVICE_ENDPOINT has been exported
    :param str s3_credentials: Path to file containing S3 credentials
    """
    os.makedirs(output_folder, exist_ok=True)
    prediction_instances = int(prediction_instances)
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if s3 is not None:
        bucket_name, service_endpoint, credentials = upload_to_s3.check_s3_credentials(s3_bucket_name, s3_service_endpoint, s3_credentials)

        input_path, fs = upload_to_s3.get_s3_path(input_path, bucket_name=bucket_name, service_endpoint=service_endpoint, credential_file=credentials)

    if slurm_task_id is not None:
        slurm_task_id = int(slurm_task_id)
    else:
        raise ValueError("The SLURM_ARRAY_TASK_ID is not set. Ensure that you are using the '-a' option with SBATCH.")

    if not os.path.isdir(os.path.join(output_folder, "mask.zarr")):
        find_mask(input_path, input_key, output_folder, s3=s3)

    # get pre-computed mean and standard deviation of full volume from JSON file
    if os.path.isfile(os.path.join(output_folder, "mean_std.json")):
        with open(os.path.join(output_folder, "mean_std.json")) as f:
            d = json.load(f)
            mean = float(d["mean"])
            std = float(d["std"])
    else:
        mean = None
        std = None

    original_shape = prediction_impl(
        input_path, input_key, output_folder, model_path, scale, block_shape, halo,
        prediction_instances=prediction_instances, slurm_task_id=slurm_task_id,
        mean=mean, std=std, s3=s3,
    )


# does NOT need GPU, FIXME: only run on CPU
def run_unet_segmentation_slurm(output_folder, min_size):
    min_size = int(min_size)
    pmap_out = os.path.join(output_folder, "predictions.zarr")
    segmentation_impl(pmap_out, output_folder, min_size=min_size)
