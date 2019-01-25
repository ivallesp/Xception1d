from unittest import TestCase
import torch
from torch import nn

from src.pytorch_utils import get_number_of_parameters


class TestGetNumberOfParameters(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_linear_model(self):
        model = nn.Linear(256, 5)
        n_params = get_number_of_parameters(model)
        self.assertEqual(n_params, 256*5+5)

    def test_mlp_model(self):
        model = nn.Sequential(nn.Linear(256, 256),
                              nn.ReLU(True),
                              nn.Linear(256, 64),
                              nn.ReLU(True),
                              nn.Linear(64, 1),
                              nn.Sigmoid())
        n_params = get_number_of_parameters(model)
        self.assertEqual(n_params, 256 * 256 + 256 + 256 * 64 + 64 + 64 * 1 + 1)

    def test_cnn(self):
        class Arch(nn.Module):
            def __init__(self):
                super(Arch, self).__init__()
                self.model_cnn = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=2, padding=1),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
                                              nn.ReLU(True),
                                              nn.MaxPool2d(3))
                self.model_linear = nn.Sequential(nn.Linear(128*10, 3),
                                                  nn.Sigmoid())

        model = Arch()
        n_params = get_number_of_parameters(model)
        self.assertEqual(n_params, 3*5*5*128 + 128 +
                                   128*3*3*128 + 128 +
                                   128*3*3*128 + 128 +
                                   128*10*3 + 3)