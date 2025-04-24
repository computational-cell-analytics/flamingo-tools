import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import torch
import z5py
from torch_em.model import UNet3d


class TestUnetPrediction(unittest.TestCase):
    shape = (64, 128, 128)

    def _create_model(self, tmp_dir):
        model = UNet3d(in_channels=1, out_channels=3, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)
        return model_path

    def _create_data(self, tmp_dir, use_tif):
        data = np.random.randint(0, 255, size=self.shape)
        if use_tif:
            path = os.path.join(tmp_dir, "data.tif")
            key = None
            imageio.imwrite(path, data)
        else:
            path = os.path.join(tmp_dir, "data.n5")
            key = "data"
            with z5py.File(path, "a") as f:
                f.create_dataset(key, data=data, chunks=(32, 32, 32))
        return path, key

    def _test_run_unet_prediction(self, use_tif, use_mask):
        from flamingo_tools.segmentation import run_unet_prediction

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path, input_key = self._create_data(tmp_dir, use_tif)
            model_path = self._create_model(tmp_dir)
            output_folder = os.path.join(tmp_dir, "output")
            run_unet_prediction(
                input_path, input_key, output_folder, model_path,
                scale=None, min_size=100,
                block_shape=(64, 64, 64), halo=(16, 16, 16),
            )

            expected_path = os.path.join(output_folder, "segmentation.zarr")
            expected_key = "segmentation"

            self.assertTrue(os.path.exists(expected_path))
            with z5py.File(expected_path, "r") as f:
                self.assertTrue(expected_key in f)
                self.assertEqual(f[expected_key].shape, self.shape)

    def test_run_unet_prediction_n5(self):
        self._test_run_unet_prediction(use_tif=False, use_mask=False)

    def test_run_unet_prediction_n5_mask(self):
        self._test_run_unet_prediction(use_tif=False, use_mask=True)

    def test_run_unet_prediction_tif(self):
        self._test_run_unet_prediction(use_tif=True, use_mask=False)

    def test_run_unet_prediction_tif_mask(self):
        self._test_run_unet_prediction(use_tif=True, use_mask=True)


if __name__ == "__main__":
    unittest.main()
