import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import argparse
import os
import cv2

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from network.classifier import *
from network.transform import mesonet_data_transforms


class MesoTrainer(object):
    def __init__(self):
        self.name = 'Mesonet'
        self.batch_size = 64
        self.epochs = 50
        self.model_name = 'meso4.pkl'
        self.model_path = './output/Mesonet/best.pkl'

        torch.backends.cudnn.benchmark = True
        self.output_path = os.path.join('./output', self.name)

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.criterion = nn.CrossEntropyLoss()
        self.model = Meso4()
        self.model = self.model.cuda()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.001,
            betas=(0.9, 0.999), eps=1e-08
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

    def train(self):
        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        iteration = 0