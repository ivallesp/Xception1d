import os
from unittest import TestCase

from src.common_paths import get_dataset_filepath, get_training_data_path
from src.data_processing import download_dataset, decompress_dataset


class TestDataDownload(TestCase):
    def test_download_and_decompress_data(self):
        data_version = "0.02"
        filepath = get_dataset_filepath(data_version=data_version)
        if os.path.exists(filepath):
            self.assertTrue(True)  # skip
        else:  # this will run in Travis
            download_dataset(data_version=data_version)
            self.assertTrue(os.path.exists(filepath))
            decompress_dataset(data_version=data_version)
            self.assertLess(10, len(os.listdir(get_training_data_path(data_version=data_version))))
