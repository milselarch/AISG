import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import torchvision

from overrides import overrides

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


class AbstractBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = None

    @property
    def weight(self):
        return self.batch_norm.weight

    @property
    def bias(self):
        return self.batch_norm.bias


class Norm1d(AbstractBatchNorm):
    def __init__(self, *args, is_training=True, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(*args, **kwargs)
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


class Norm2d(AbstractBatchNorm):
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


class Norm3d(AbstractBatchNorm):
    def __init__(self, *args, is_training=True, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(*args, **kwargs)
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
