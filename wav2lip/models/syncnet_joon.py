#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch

from torch import nn
from torch.nn import functional as F
from .CustomBatchNorms import *

def save(model, filename):
    with open(filename, "wb") as f:
        torch.save(model, f)
        print("%s saved." % filename)

def load(filename):
    net = torch.load(filename)
    return net

class SyncnetJoon(nn.Module):
    def __init__(
        self, num_layers_in_fc_layers=1024,
        fcc_ratio=0., fcc_list=(128, 16)
    ):
        super(SyncnetJoon, self).__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32

        self.fcc_ratio = fcc_ratio
        self.train_var = BooleanVar(True)

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=(3, 3), stride=(1, 1),
                padding=(1, 1)
            ),
            Norm2d(64, is_training=self.train_var),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(
                64, 192, kernel_size=(3, 3), stride=(1, 1),
                padding=(1, 1)
            ),
            Norm2d(192, is_training=self.train_var),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            Norm2d(384, is_training=self.train_var),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            Norm2d(256, is_training=self.train_var),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            Norm2d(256, is_training=self.train_var),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            Norm2d(512, is_training=self.train_var),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            Norm1d(512, is_training=self.train_var),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            Norm1d(512, is_training=self.train_var),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(
                3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                padding=0
            ),
            Norm3d(96, is_training=self.train_var),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(
                96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2),
                padding=(0, 1, 1)
            ),
            Norm3d(256, is_training=self.train_var),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3), stride=(1, 2, 2),
                padding=(0, 1, 1)
            ),

            nn.Conv3d(
                256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            Norm3d(256, is_training=self.train_var),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            Norm3d(256, is_training=self.train_var),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            Norm3d(256, is_training=self.train_var),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(
                256, 512, kernel_size=(1, 6, 6), padding=0
            ),
            Norm3d(512, is_training=self.train_var),
            nn.ReLU(inplace=True),
        )

        self.dense_layer = self.make_dense(fcc_list)

    @staticmethod
    def make_dense(fcc_list):
        if type(fcc_list) is int:
            fcc_list = (fcc_list,)

        num_neurons = fcc_list[0]
        dense_layers = []

        for k in range(1, len(fcc_list)):
            num_neurons = fcc_list[k]
            prev_neurons = fcc_list[k-1]
            dense_layers.extend([
                nn.Linear(prev_neurons, num_neurons),
                nn.ReLU()
            ])

        dense_sequential = nn.Sequential(
            nn.Linear(2048, fcc_list[0]),
            nn.ReLU(),
            *dense_layers,
            nn.Linear(num_neurons, 1)
        )

        return dense_sequential

    @overrides
    def train(self, mode: bool = True):
        super().train(mode)
        self.train_var.set(mode)

    def forward(self, audio_sequences, face_sequences):
        aud_out = self.forward_aud(audio_sequences)
        lip_out = self.forward_lip(face_sequences)
        return aud_out, lip_out

    def predict_distance(self, *args, **kwargs):
        audio_embed, face_embed = self.forward(*args, **kwargs)
        # print('EMBEDS', audio_embed.shape, face_embed.shape)
        distance = torch.nn.functional.pairwise_distance(
            audio_embed, face_embed
        )

        # print('SIM JOON =', similarity)
        return distance

    def predict(self, *args, **kwargs):
        audio_embed, face_embed = self.forward(*args, **kwargs)
        # print('EMBEDS', audio_embed.shape, face_embed.shape)
        audio_embed = F.normalize(audio_embed, p=2, dim=1)
        face_embed = F.normalize(face_embed, p=2, dim=1)

        cosine_similarity = nn.functional.cosine_similarity
        similarity = cosine_similarity(audio_embed, face_embed)
        similarity = (similarity + 1.) / 2.
        return 1.0 - similarity

    def forward_aud(self, x):
        mid = self.netcnnaud(x)  # N x ch x 24 x M
        mid = mid.view((mid.size()[0], -1))  # N x (ch x 24)
        out = self.netfcaud(mid)
        return out

    def forward_lip(self, x):
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))  # N x (ch x 24)
        out = self.netfclip(mid)
        return out

    def forward_lipfeat(self, x):
        mid = self.netcnnlip(x)
        out = mid.view((mid.size()[0], -1))  # N x (ch x 24)
        return out

    def dense_predict(
        self, audio_embeds, image_embeds, sigmoid=True
    ):
        flat_embeds = torch.cat(
            [image_embeds, audio_embeds], dim=-1
        )

        prediction = self.dense_layer(flat_embeds)
        if sigmoid:
            prediction = torch.sigmoid(prediction)

        return prediction

    def pred_fcc(self, audios, images, fcc_ratio=None, sigmoid=True):
        if fcc_ratio is None:
            fcc_ratio = self.fcc_ratio

        audio_embeds, image_embeds = self.forward(audios, images)
        prediction = self.dense_predict(
            audio_embeds, image_embeds, sigmoid=sigmoid
        )

        if fcc_ratio == 1.:
            return prediction

        cosine_similarity = nn.functional.cosine_similarity
        similarity = cosine_similarity(audio_embeds, face_embeds)
        cosine_fakeness = 1.0 - similarity
        prediction = (
            (1-fcc_ratio) * cosine_fakeness +
            fcc_ratio * prediction
        )

        return prediction