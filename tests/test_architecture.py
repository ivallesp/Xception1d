from unittest import TestCase

import numpy as np
import torch
from scipy.io import wavfile

from src.architecture import XceptionArchitecture1d


class TestXceptionArchitecture1d(TestCase):
    def setUp(self):
        self.n_classes = 32
        self.batch_size = 16
        self.x = torch.rand(self.batch_size, 1, 16000)
        self.y = torch.from_numpy(np.random.choice(range(self.n_classes), self.batch_size))
        self.model = XceptionArchitecture1d(n_classes=self.n_classes)

    def tearDown(self):
        pass

    def test_forward_pass_shape(self):
        y_hat = self.model.forward(self.x)
        self.assertEqual(y_hat.shape, (self.batch_size, self.n_classes))

    def test_loss(self):
        loss, _ = self.model.calculate_loss(self.x, self.y)
        self.assertGreater(loss, 0)

    # def test_gradient_check(self):
    #     loss_0, _ = self.model.step(self.x, self.y)
    #     loss_1, _ = self.model.calculate_loss(self.x, self.y)
    #     self.assertGreater(loss_0, loss_1)

    def test_forward_pass_with_real_wav_file(self):
        sr, wav = wavfile.read("./tests/examples/testaudio.wav")
        wav = np.pad(wav, int((16000-len(wav))/2), "constant")
        wav = np.expand_dims(np.expand_dims(wav, 0),1)
        wav = torch.from_numpy(wav).float()
        self.model.eval()
        output = self.model.forward(wav)
        self.model.train()
        # No error means test has passed
