import numpy as np
import pandas as pd
import torch
import zarr

from skimage.filters import gaussian
from torch_em.util import ensure_tensor_with_channels


# Process labels stored in json napari style.
# I don't actually think that we need the epsilon here, but will leave it for now.
def process_labels(label_path, shape, sigma, eps):
    labels = np.zeros(shape, dtype="float32")
    points = pd.read_csv(label_path)
    assert len(points.columns) == len(shape)
    coords = tuple(
        np.clip(np.round(points[ax].values).astype("int"), 0, shape[i] - 1)
        for i, ax in enumerate(points.columns)
    )
    labels[coords] = 1
    labels = gaussian(labels, sigma)
    # TODO better normalization?
    labels /= labels.max()
    return labels


class DetectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    def __init__(
        self,
        raw_image_paths,
        label_paths,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
        eps=1e-8,
        sigma=None,
        **kwargs,
    ):
        self.raw_images = raw_image_paths
        # TODO make this a parameter
        self.raw_key = "raw"
        self.label_images = label_paths
        self._ndim = 3

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        self.eps = eps
        self.sigma = sigma

        if n_samples is None:
            self._len = len(self.raw_images)
            self.sample_random_index = False
        else:
            self._len = n_samples
            self.sample_random_index = True

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, shape):
        if any(sh < psh for sh, psh in zip(shape, self.patch_shape)):
            raise NotImplementedError(
                f"Image padding is not supported yet. Data shape {shape}, patch shape {self.patch_shape}"
            )
        bb_start = [
            np.random.randint(0, sh - psh) if sh - psh > 0 else 0
            for sh, psh in zip(shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_sample(self, index):
        if self.sample_random_index:
            index = np.random.randint(0, len(self.raw_images))
        raw, label = self.raw_images[index], self.label_images[index]

        raw = zarr.open(raw)[self.raw_key]
        # Note: this is quite inefficient, because we process the full crop rather than
        # just the requested bounding box.
        label = process_labels(label, raw.shape, self.sigma, self.eps)

        have_raw_channels = raw.ndim == 4  # 3D with channels
        have_label_channels = label.ndim == 4
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        shape = raw.shape
        prefix_box = tuple()
        if have_raw_channels:
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        bb = self._sample_bounding_box(shape)
        raw_patch = np.array(raw[prefix_box + bb])
        label_patch = np.array(label[bb])

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, label_patch):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                label_patch = np.array(label[bb])
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        if have_raw_channels and len(prefix_box) == 0:
            raw_patch = raw_patch.transpose((3, 0, 1, 2))  # Channels, Depth, Height, Width

        return raw_patch, label_patch

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        # initial_label_dtype = labels.dtype

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels


if __name__ == "__main__":
    import napari

    raw_path = "training_data/images/10.1L_mid_IHCribboncount_5_Z.zarr"
    label_path = "training_data/labels/10.1L_mid_IHCribboncount_5_Z.csv"

    f = zarr.open(raw_path, "r")
    raw = f["raw"][:]

    labels = process_labels(label_path, shape=raw.shape, sigma=1, eps=1e-7)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(labels)
    napari.run()
