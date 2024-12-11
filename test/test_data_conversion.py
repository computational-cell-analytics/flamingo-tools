import os
import unittest
from shutil import rmtree


class TestDataConversion(unittest.TestCase):
    folder = "./tmp"

    def setUp(self):
        from flamingo_tools import create_test_data

        # TODO Create flamingo metadata.
        create_test_data(self.folder)

    def tearDown(self):
        rmtree(self.folder)

    def test_convert_lightsheet_to_bdv(self):
        from flamingo_tools import convert_lightsheet_to_bdv

        out_path = os.path.join(self.folder, "converted_data.n5")
        convert_lightsheet_to_bdv(self.folder, out_path=out_path, metadata_file_name_pattern=None)

        self.assertTrue(os.path.exists(out_path))
        xml_path = out_path.replace(".n5", ".xml")
        self.assertTrue(os.path.exists(xml_path))


if __name__ == "__main__":
    unittest.main()
