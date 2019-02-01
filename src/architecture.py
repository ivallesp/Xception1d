import torch
from torch import nn

from src.pytorch_modules import XceptionModule1d, Flatten, DepthwiseSeparableConv1d, Swish


class XceptionArchitecture1d(nn.Module):
    def __init__(self, n_classes, lr=2.5e-4, bn_momentum=0.995):
        # Initialization
        # Remember not to add the activation at the end
        super(XceptionArchitecture1d, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.init_flow = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32,
                                                 stride=4, padding=4, kernel_size=9),
                                       Swish(),
                                       nn.BatchNorm1d(32, momentum=bn_momentum),
                                       nn.Conv1d(in_channels=32, out_channels=64,
                                                 stride=2, padding=4, kernel_size=5))

        # Entry flow
        self.entry_flow = nn.Sequential(XceptionModule1d(in_channels=64, out_channels=128,
                                                         n_modules=2, kernel_size=3, pooling_stride=2),
                                        XceptionModule1d(in_channels=128, out_channels=256,
                                                         n_modules=2, kernel_size=3, pooling_stride=2),
                                        XceptionModule1d(in_channels=256, out_channels=728,
                                                         n_modules=2, kernel_size=3, pooling_stride=2))

        # Middle flow
        modules = [XceptionModule1d(in_channels=728, out_channels=728,
                                    n_modules=3, kernel_size=3, pooling_stride=1) for _ in range(8)]
        self.middle_flow = nn.Sequential(*modules)

        # Exit flow
        self.exit_flow = nn.Sequential(XceptionModule1d(in_channels=728, out_channels=1024,
                                                        n_modules=2, kernel_size=3, pooling_stride=2),
                                       Swish(),
                                       nn.BatchNorm1d(1024, momentum=bn_momentum),
                                       DepthwiseSeparableConv1d(in_channels=1024, out_channels=1536, kernel_size=3,
                                                                stride=2),
                                       Swish(),
                                       nn.BatchNorm1d(1536, momentum=bn_momentum),
                                       DepthwiseSeparableConv1d(in_channels=1536, out_channels=2048, kernel_size=3,
                                                                stride=2))

        # FC flow
        self.fc_flow = nn.Sequential(Flatten(),
                                     Swish(),
                                     nn.BatchNorm1d(2048 * 32, momentum=bn_momentum),
                                     nn.Dropout(p=0.75, inplace=True),
                                     nn.Linear(2048*32, n_classes))

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x):
        h = self.init_flow.forward(x)
        h = self.entry_flow.forward(h)
        h = self.middle_flow.forward(h)
        h = self.exit_flow.forward(h)
        h = self.fc_flow(h)
        return h

    def calculate_loss(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.squeeze().long())
        return loss, y_hat

    def step(self, x, y):
        loss, y_hat = self.calculate_loss(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, y_hat
