from unittest import TestCase
import torch

from src.pytorch_modules import DepthwiseSeparableConv2d, DepthwiseSeparableConv1d


class TestDepthwiseSeparableConv2d(TestCase):
    def setUp(self):
        self.input_tensor = torch.rand(128, 3, 25, 25)

    def tearDown(self):
        pass

    def test_shapes(self):
        conv_op = DepthwiseSeparableConv2d(in_channels=3,
                                           out_channels=64)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 25, 25))

    def test_gradient_check(self):
        conv_op = DepthwiseSeparableConv2d(in_channels=3,
                                           out_channels=64)
        for i in range(10):
            loss = conv_op.forward(self.input_tensor).sum()
            opt = torch.optim.SGD(params=conv_op.parameters(), lr=1e-5)
            opt.zero_grad()
            loss.backward()
            opt.step()
        new_loss = conv_op.forward(self.input_tensor).sum()
        self.assertGreater(loss, new_loss)


class TestDepthwiseSeparableConv1d(TestCase):
    def setUp(self):
        self.input_tensor = torch.rand(128, 3, 25)

    def tearDown(self):
        pass

    def test_shapes(self):
        conv_op = DepthwiseSeparableConv1d(in_channels=3,
                                           out_channels=64)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 25))

    def test_gradient_check(self):
        conv_op = DepthwiseSeparableConv1d(in_channels=3,
                                           out_channels=64)
        for i in range(10):
            loss = conv_op.forward(self.input_tensor).sum()
            opt = torch.optim.SGD(params=conv_op.parameters(), lr=1e-5)
            opt.zero_grad()
            loss.backward()
            opt.step()
        new_loss = conv_op.forward(self.input_tensor).sum()
        self.assertGreater(loss, new_loss)
