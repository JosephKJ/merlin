# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/GradientEpisodicMemory

from lib.config import cfg
from lib.baselines.common import MLP, ResNet18

import torch


class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks):
        super(Net, self).__init__()
        nl = 2      # Number of layers
        nh = 100    # Number of hidden neurons

        # Setup network
        self.is_cifar = 'cifar' in cfg.continual.task or 'mini_imagenet' in cfg.continual.task
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        # Setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=cfg.continual.learning_rate)

        # Setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        self.train()
        self.zero_grad()
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            self.bce((self.net(x)[:, offset1: offset2]),
                     y - offset1).backward()
        else:
            self.bce(self(x, t), y).backward()
        self.opt.step()
