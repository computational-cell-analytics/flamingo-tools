import os
import tempfile
import unittest

from elf.io import open_file
from skimage.data import binary_blobs
from skimage.measure import label


class TestPostprocessing(unittest.TestCase):
    def _create_example_seg(self, tmp_dir):
        seg = binary_blobs(256, n_dim=3, volume_fraction=0.2)
        seg = label(seg)
        return seg

    def _test_postprocessing(self, spatial_statistics, threshold, **spatial_statistics_kwargs):
        from flamingo_tools.segmentation.postprocessing import filter_segmentation

        with tempfile.TemporaryDirectory() as tmp_dir:
            example_seg = self._create_example_seg(tmp_dir)
            output_path = os.path.join(tmp_dir, "test-output.zarr")
            output_key = "seg-filtered"
            filter_segmentation(
                example_seg, output_path, spatial_statistics, threshold,
                output_key=output_key, **spatial_statistics_kwargs
            )
            self.assertTrue(os.path.exists(output_path))
            with open_file(output_path, "r") as f:
                filtered_seg = f[output_key][:]
            self.assertEqual(filtered_seg.shape, example_seg.shape)

    def test_nearest_neighbor_distance(self):
        from flamingo_tools.segmentation.postprocessing import nearest_neighbor_distance

        self._test_postprocessing(nearest_neighbor_distance, threshold=5)

    def test_local_ripleys_k(self):
        from flamingo_tools.segmentation.postprocessing import local_ripleys_k

        self._test_postprocessing(local_ripleys_k, threshold=0.5)

    def test_neighbors_in_radius(self):
        from flamingo_tools.segmentation.postprocessing import neighbors_in_radius

        self._test_postprocessing(neighbors_in_radius, threshold=5)


if __name__ == "__main__":
    unittest.main()
