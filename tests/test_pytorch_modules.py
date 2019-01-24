from unittest import TestCase
import torch

from src.pytorch_modules import DepthwiseSeparableConv2d, DepthwiseSeparableConv1d, TransferenceFunctionModule, \
    XceptionModule1d


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


class TestTransferenceFunctionModule(TestCase):
    def setUp(self):
        self.input_tensor = torch.rand(128, 3, 25)

    def tearDown(self):
        pass

    def test_shapes(self):
        conv_op = TransferenceFunctionModule()
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 3, 25))

    def test_equal(self):
        conv_op = TransferenceFunctionModule()
        output = conv_op.forward(self.input_tensor)
        self.assertTrue((self.input_tensor == output).numpy().all())


class TestXceptionModule1d(TestCase):
    def setUp(self):
        self.input_tensor = torch.rand(128, 3, 25)

    def tearDown(self):
        pass

    def test_shapes_single_module(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 1,
                                    kernel_size = 3,
                                    pooling_stride = 1)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 25))

    def test_shapes_multiple_modules(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 3,
                                    pooling_stride = 1)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 25))

    def test_shapes_with_pooling(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 3,
                                    pooling_stride = 2)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 13))

    def test_shapes_multiple_modules_with_pooling(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 3,
                                    pooling_stride = 2)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 13))

    def test_shapes_multiple_modules_big_size(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 9,
                                    pooling_stride = 1)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 25))


    def test_shapes_multiple_modules_big_stride(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 3,
                                    pooling_stride = 5)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 5))

    def test_shapes_multiple_modules_big_stride_indivisible(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 3,
                                    pooling_stride = 4)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 7))

    def test_shapes_multiple_modules_big_stride_indivisible_and_big_size(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 9,
                                    pooling_stride = 4)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 7))

        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 5,
                                    kernel_size = 9,
                                    pooling_stride = 12)
        output = conv_op.forward(self.input_tensor)
        self.assertEqual(output.shape, (128, 64, 3))

    def test_gradient_check(self):
        conv_op = XceptionModule1d(in_channels=3,
                                    out_channels=64,
                                    n_modules = 1,
                                    kernel_size = 3,
                                    pooling_stride = 2)
        for i in range(10):
            loss = conv_op.forward(self.input_tensor).sum()
            opt = torch.optim.SGD(params=conv_op.parameters(), lr=1e-5)
            opt.zero_grad()
            loss.backward()
            opt.step()
        new_loss = conv_op.forward(self.input_tensor).sum()
        self.assertGreater(loss, new_loss)