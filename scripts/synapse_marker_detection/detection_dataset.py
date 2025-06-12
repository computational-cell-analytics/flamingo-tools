import numpy as np
import pandas as pd
import torch
import zarr

from skimage.filters import gaussian
from torch_em.util import ensure_tensor_with_channels


class MinPointSampler:
    """A sampler to reject samples with a low fraction of foreground pixels in the labels.

    Args:
        min_fraction: The minimal fraction of foreground pixels for accepting a sample.
        background_id: The id of the background label.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, min_points: int, p_reject: float = 1.0):
        self.min_points = min_points
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, n_points: int) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data.

        Returns:
            Whether to accept this sample.
        """

        if n_points > self.min_points:
            return True
        else:
            return np.random.rand() > self.p_reject


def load_labels(label_path, shape, bb):
    points = pd.read_csv(label_path)
    assert len(points.columns) == len(shape)
    z_coords, y_coords, x_coords = points["axis-0"].values, points["axis-1"].values, points["axis-2"].values

    if bb is not None:
        (z_min, z_max), (y_min, y_max), (x_min, x_max) = [(s.start, s.stop) for s in bb]
        z_coords -= z_min
        y_coords -= y_min
        x_coords -= x_min
        mask = np.logical_and.reduce([
            np.logical_and(z_coords >= 0, z_coords < (z_max - z_min)),
            np.logical_and(y_coords >= 0, y_coords < (y_max - y_min)),
            np.logical_and(x_coords >= 0, x_coords < (x_max - x_min)),
        ])
        z_coords, y_coords, x_coords = z_coords[mask], y_coords[mask], x_coords[mask]
        restricted_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        shape = restricted_shape

    n_points = len(z_coords)
    coords = tuple(
        np.clip(np.round(coord).astype("int"), 0, coord_max - 1) for coord, coord_max in zip(
            (z_coords, y_coords, x_coords), shape
        )
    )

    return coords, n_points


# Process labels stored in json napari style.
# I don't actually think that we need the epsilon here, but will leave it for now.
def process_labels(coords, shape, sigma, eps, bb=None):

    if bb:
        (z_min, z_max), (y_min, y_max), (x_min, x_max) = [(s.start, s.stop) for s in bb]
        restricted_shape = (z_max - z_min, y_max - y_min, x_max - x_min)
        labels = np.zeros(restricted_shape, dtype="float32")
        shape = restricted_shape
    else:
        labels = np.zeros(shape, dtype="float32")

    labels[coords] = 1
    labels = gaussian(labels, sigma)
    # TODO better normalization?
    labels /= (labels.max() + 1e-7)
    labels *= 4
    return labels


class DetectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    @staticmethod
    def compute_len(shape, patch_shape):
        if patch_shape is None:
            return 1
        else:
            n_samples = int(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
            return n_samples

    def __init__(
        self,
        raw_path,
        label_path,
        patch_shape,
        raw_key,
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
        self.raw_path = raw_path
        self.label_path = label_path
        self.raw_key = raw_key
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

        with zarr.open(self.raw_path, "r") as f:
            self.shape = f[self.raw_key].shape

        if n_samples is None:
            self._len = self.compute_len(self.shape, self.patch_shape) if n_samples is None else n_samples
        else:
            self._len = n_samples

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
        raw, label_path = self.raw_path, self.label_path

        raw = zarr.open(raw)[self.raw_key]
        have_raw_channels = raw.ndim == 4  # 3D with channels
        shape = raw.shape

        bb = self._sample_bounding_box(shape)
        prefix_box = tuple()
        if have_raw_channels:
            if shape[-1] < 16:
                shape = shape[:-1]
            else:
                shape = shape[1:]
                prefix_box = (slice(None), )

        raw_patch = np.array(raw[prefix_box + bb])

        coords, n_points = load_labels(label_path, shape, bb)
        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw_patch, n_points):
                bb = self._sample_bounding_box(shape)
                raw_patch = np.array(raw[prefix_box + bb])
                coords, n_points = load_labels(label_path, shape, bb)
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(f"Could not sample a valid batch in {self.max_sampling_attempts} attempts")

        label = process_labels(coords, shape, self.sigma, self.eps, bb=bb)

        have_label_channels = label.ndim == 4
        if have_label_channels:
            raise NotImplementedError("Multi-channel labels are not supported.")

        label_patch = np.array(label)

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
