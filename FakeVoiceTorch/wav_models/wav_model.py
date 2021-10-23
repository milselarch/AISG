import torch
from torch import nn
from torch.nn import functional as F

from . import wav2lip

def dense(inputs, outputs, dropout=0.5, neg_slope=-0.1):
    return nn.Sequential(
        nn.Linear(inputs, outputs),
        nn.LeakyReLU(negative_slope=neg_slope),
        nn.Dropout(p=dropout)
    )

class WavDisc(nn.Module):
    def __init__(self, fcc=(512, 128, 32)):
        super(WavDisc, self).__init__()

        # torch.Size([128, 512, 1, 1])
        self.markers = []
        mark = self.mark

        self.wav2lip = mark(wav2lip.Wav2Lip())
        self.fcc = fcc

        dense_block = []
        for k in range(len(self.fcc) - 1):
            inputs, outputs = self.fcc[k], self.fcc[k + 1]
            layer = dense(inputs, outputs)
            dense_block.append(layer)

        self.dense_block = mark(nn.Sequential(*dense_block))
        self.final_dense = mark(dense(self.fcc[-1], 1))

    def mark(self, tensor):
        # assert isinstance(tensor, torch.Tensor)
        self.markers.append(tensor)
        return tensor

    def to_cuda(self):
        super().cuda()
        for tensor in self.markers:
            # assert isinstance(tensor, torch.Tensor)
            tensor.cuda()

    def forward(self, audio_inputs):
        encodings = self.wav2lip.encode_audio(audio_inputs)

        while encodings.shape[-2] == 1:
            encodings = torch.squeeze(encodings, -2)
            encodings = torch.squeeze(encodings, -1)

        output = self.dense_block(encodings)
        output = self.final_dense(output)
        output = torch.sigmoid(output)
        return output
