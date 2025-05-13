import os
import unittest
from shutil import rmtree

import imageio.v3 as imageio
import pandas as pd
import numpy as np
from skimage.measure import regionprops_table


class TestMeasurements(unittest.TestCase):
    folder = "./tmp"

    def setUp(self):
        from flamingo_tools.test_data import get_test_volume_and_segmentation

        self.image_path, self.seg_path, self.table_path = get_test_volume_and_segmentation(self.folder)

    def tearDown(self):
        try:
            rmtree(self.folder)
        except Exception:
            pass

    def test_compute_object_measures(self):
        from flamingo_tools.measurements import compute_object_measures

        output_path = os.path.join(self.folder, "measurements.tsv")
        compute_object_measures(
            self.image_path, self.seg_path, self.table_path, output_path, n_threads=1
        )
        self.assertTrue(os.path.exists(output_path))

        table = pd.read_csv(output_path, sep="\t")
        self.assertTrue(len(table) >= 1)
        expected_columns = ["label_id", "mean", "stdev", "min", "max", "median"]
        expected_columns.extend([f"percentile-{p}" for p in (5, 10, 25, 75, 90, 95)])
        expected_columns.extend(["volume", "surface"])
        for col in expected_columns:
            self.assertIn(col, table.columns)

        n_objects = int(imageio.imread(self.seg_path).max())
        expected_shape = (n_objects, len(expected_columns))
        self.assertEqual(table.shape, expected_shape)

        image = imageio.imread(self.image_path)
        segmentation = imageio.imread(self.seg_path)
        properties = ("label", "intensity_mean", "intensity_std", "intensity_min", "intensity_max")
        expected_measures = regionprops_table(segmentation, intensity_image=image, properties=properties)
        expected_measures = pd.DataFrame(expected_measures)

        for (col, col_exp) in [
            ("label_id", "label"), ("mean", "intensity_mean"), ("stdev", "intensity_std"),
            ("min", "intensity_min"), ("max", "intensity_max"),
        ]:
            self.assertTrue(np.allclose(table[col].values, expected_measures[col_exp].values))


if __name__ == "__main__":
    unittest.main()
