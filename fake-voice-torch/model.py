import pytorch
import pytorch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self, num_freq_bin, init_neurons
    ):
        self.conv1 = self.causal_conv_1d(
            in_channels=num_freq_bin, kernel_size=3,
            stride=1, out_channels=init_neurons
        )

    @classmethod
    def causal_conv_1d(
        cls, in_channels, out_channels, kernel_size,
        dilation=1, **kwargs
    ):
        pad = (kernel_size - 1) * dilation

        return nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=pad, dilation=dilation, **kwargs
        )

    def forward(self, input_values):


