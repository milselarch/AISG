import torch

from torch import nn
from torch.nn import functional as F

class CollateModel(nn.Module):
    def __init__(
        self, dropout_p=0.5, num_outputs=1,
        num_inputs=2, fcc_list=(16, 32, 16, 8)
    ):
        super(CollateModel, self).__init__()

        self.fcc_list = fcc_list
        self.dropout_p = dropout_p
        self.num_outputs = num_outputs
        self.model = self.make_dense(
            fcc_list=fcc_list, dropout_p=dropout_p,
            num_outputs=num_outputs, num_inputs=num_inputs
        )

    def forward(self, inputs, sigmoid=True):
        output = self.model(inputs)
        output = torch.sigmoid(output) if sigmoid else output
        if self.num_outputs == 1:
            output = torch.flatten(output)

        return output

    @staticmethod
    def make_dense(
        fcc_list, dropout_p=0.0, num_outputs=1,
        num_inputs=2
    ):
        if type(fcc_list) is int:
            fcc_list = (fcc_list,)

        num_neurons = fcc_list[0]
        dense_layers = [
            nn.Linear(num_inputs, fcc_list[0]),
        ]

        if dropout_p > 0.0:
            dense_layers.append(nn.Dropout(
                dropout_p
            ))

        dense_layers.append(nn.ReLU())

        for k in range(1, len(fcc_list)):
            num_neurons = fcc_list[k]
            prev_neurons = fcc_list[k - 1]
            dense_layers.append(nn.Linear(prev_neurons, num_neurons))

            if dropout_p > 0.0:
                dense_layers.append(nn.Dropout(dropout_p))

            dense_layers.append(nn.ReLU())

        dense_sequential = nn.Sequential(
            *dense_layers,
            nn.Linear(num_neurons, num_outputs)
        )

        return dense_sequential