import os
from unittest import TestCase

from src.common_paths import get_dataset_filepath, get_training_data_path
from src.data_processing import download_dataset, decompress_dataset


class TestDataDownload(TestCase):
    def test_download_and_decompress_data(self):
        filepath = get_dataset_filepath()
        if os.path.exists(filepath):
            self.assertTrue(True)  # skip
        else:  # this will run in Travis
            download_dataset()
            self.assertTrue(os.path.exists(filepath))
            decompress_dataset()
            self.assertLess(10, len(os.listdir(get_training_data_path())))
