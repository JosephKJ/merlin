import torch.nn as nn
import torch.nn.functional as F
import math


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class SimpleNet_3x3(nn.Module):
    def __init__(self):
        super(SimpleNet_3x3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3)
        self.mp2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.fc = nn.Linear(50 * 4 * 4, 10)

    def forward(self, x):
        x = self.mp1(F.leaky_relu(self.conv1(x)))
        x = self.mp2(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 50 * 4 * 4)
        x = self.fc(x)
        return x


class SimpleNet_FC(nn.Module):
    def __init__(self):
        nl = 2  # Number of layers
        nh = 50  # Number of hidden neurons
        sizes = [784] + [nh] * nl + [10]
        super(SimpleNet_FC, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


class SimpleNet_FC_100(nn.Module):
    def __init__(self):
        nl = 2  # Number of layers
        nh = 100  # Number of hidden neurons
        sizes = [784] + [nh] * nl + [10]
        super(SimpleNet_FC_100, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)
