import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import torchvision

from overrides import overrides


class CustomBatchNorm2d(nn.Module):
    def __init__(self, *args, is_training=True, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(*args, **kwargs)
        self.is_training = is_training

    def __call__(self, *args, **kwargs):
        if self.is_training:
            return self.batch_norm(*args, **kwargs)

        # eval mode messes up batch norm real bad
        state = self.batch_norm.state_dict()
        state = copy.deepcopy(state)

        self.train(True)
        norm_output = self.batch_norm(*args, **kwargs)
        self.train(False)
        self.batch_norm.load_state_dict(state)
        return norm_output


class BooleanVar(object):
    def __init__(self, value=True):
        self.value = value

    def set_true(self):
        self.value = True

    def set_false(self):
        self.value = False

    def set(self, value):
        if value:
            self.value = True
        else:
            self.value = False

    def __bool__(self):
        return self.value


class Meso4(nn.Module):
    """
    Pytorch Implemention of Meso4
    Autor: Honggu Liu
    Date: July 4, 2019
    """

    def __init__(self, num_classes=2, use_sigmoid=False):
        super(Meso4, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.is_training = BooleanVar(True)

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = CustomBatchNorm2d(
            8, is_training=self.is_training
        )
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = CustomBatchNorm2d(
            16, is_training=self.is_training
        )
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        # flatten: x = x.view(x.size(0), -1)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    @overrides
    def train(self, mode: bool = True):
        super().train(mode)
        self.is_training.set(mode)

    def forward(self, input):
        # (8, 256, 256)
        x = self.conv1(input)
        x = self.relu(x)
        x = self.bn1(x)
        # (8, 128, 128)
        x = self.maxpooling1(x)

        # (8, 128, 128)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        # (8, 64, 64)
        x = self.maxpooling1(x)

        # (16, 64, 64)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn2(x)
        # (16, 32, 32)
        x = self.maxpooling1(x)

        # (16, 32, 32)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn2(x)
        # (16, 8, 8)
        x = self.maxpooling2(x)

        # (Batch, 16*8*8)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # (Batch, 16)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x


class MesoInception4(nn.Module):
    """
    Pytorch Implemention of MesoInception4
    Author: Honggu Liu
    Date: July 7, 2019
    """

    def __init__(self, num_classes=2, use_sigmoid=False):
        super(MesoInception4, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.is_training = BooleanVar(True)
        train_var = self.is_training

        self.num_classes = num_classes
        # InceptionLayer1
        self.x1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.x1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.x1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.x1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.x1_conv3_2 = nn.Conv2d(
            4, 4, 3, padding=2, dilation=2, bias=False
        )
        self.x1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.x1_conv4_2 = nn.Conv2d(
            2, 2, 3, padding=3, dilation=3, bias=False
        )

        self.x1_bn = CustomBatchNorm2d(11, is_training=train_var)

        # InceptionLayer2
        self.x2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.x2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.x2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.x2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.x2_conv3_2 = nn.Conv2d(
            4, 4, 3, padding=2, dilation=2, bias=False
        )
        self.x2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.x2_conv4_2 = nn.Conv2d(
            2, 2, 3, padding=3, dilation=3, bias=False
        )

        self.x2_bn = CustomBatchNorm2d(12, is_training=train_var)

        # Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = CustomBatchNorm2d(16, is_training=train_var)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    @overrides
    def train(self, mode: bool = True):
        super().train(mode)
        self.is_training.set(mode)

    # InceptionLayer
    def inception_layer1(self, input):
        x1 = self.x1_conv1(input)
        x2 = self.x1_conv2_1(input)
        x2 = self.x1_conv2_2(x2)
        x3 = self.x1_conv3_1(input)
        x3 = self.x1_conv3_2(x3)
        x4 = self.x1_conv4_1(input)
        x4 = self.x1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.x1_bn(y)
        y = self.maxpooling1(y)

        return y

    def inception_layer2(self, input):
        x1 = self.x2_conv1(input)
        x2 = self.x2_conv2_1(input)
        x2 = self.x2_conv2_2(x2)
        x3 = self.x2_conv3_1(input)
        x3 = self.x2_conv3_2(x3)
        x4 = self.x2_conv4_1(input)
        x4 = self.x2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.x2_bn(y)
        y = self.maxpooling1(y)

        return y

    def forward(self, input):
        # (Batch, 11, 128, 128)
        x = self.inception_layer1(input)
        # (Batch, 12, 64, 64)
        x = self.inception_layer2(x)

        # (Batch, 16, 64 ,64)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        # (Batch, 16, 32, 32)
        x = self.maxpooling1(x)

        # (Batch, 16, 32, 32)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        # (Batch, 16, 8, 8)
        x = self.maxpooling2(x)

        # (Batch, 16*8*8)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # (Batch, 16)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x
