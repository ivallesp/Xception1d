import os
import shutil
from unittest import TestCase

from src.common_paths import get_data_path, get_models_path, get_model_path, get_outputs_path, \
    get_output_path, get_training_data_path, get_dataset_filepath, get_augmented_data_path


class TestDataTools(TestCase):
    def setUp(self):
        self.version_id = "test_version"

    def tearDown(self):
        pass

    def test_get_data_path(self):
        path = get_data_path()
        self.assertTrue(os.path.exists(path))

    def test_get_models_path(self):
        path = get_models_path()
        self.assertTrue(os.path.exists(path))

    def test_get_model_path(self):
        path = get_model_path(self.version_id)
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(path)

    def test_get_outputs_path(self):
        path = get_outputs_path()
        self.assertTrue(os.path.exists(path))

    def test_get_output_path(self):
        path = get_output_path(self.version_id)
        self.assertTrue(os.path.exists(path))
        shutil.rmtree(path)

    def test_get_training_data_path(self):
        path = get_training_data_path("unit_testing")
        self.assertTrue(os.path.exists(path))
        os.rmdir(path)

    def test_get_dataset_filepath(self):
        path = get_dataset_filepath(data_version="unit_testing")
        self.assertTrue(os.path.exists(os.path.split(path)[0]))
        self.assertTrue(path.endswith(".tar.gz"))

    def test_get_augmented_data_path(self):
        path = get_augmented_data_path(data_version="unit_testing")
        self.assertTrue(os.path.exists(path))
        os.rmdir(path)
