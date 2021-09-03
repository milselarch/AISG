import torch
import torch.nn as nn

def custom_pooling(x):
    target = x[1]
    inputs = x[0]
    mask_value = 0

    # getting the mask by observing the model's inputs
    mask = torch.eq(inputs, mask_value)
    # mask = K.all(mask, axis=-1, keepdims=True)
    mask = torch.all(mask, dim=-1, keepdim=True)

    # inverting the mask for getting the valid steps for each sample
    mask = 1 - mask.float()

    # summing the valid steps for each sample
    steps_per_sample = torch.sum(mask, dim=1, keepdim=False)

    # applying the mask to the target
    # (to make sure you are summing zeros below)
    target = target * mask

    # calculating the mean of the steps
    # (using our sum of valid steps as averager)
    total = torch.sum(target, dim=1, keepdim=False)
    means = total / steps_per_sample
    return means

class Discriminator(nn.Module):
    def __init__(
        self, num_freq_bin, init_neurons, num_conv_blocks,
        residual_con, num_dense_neurons, dense_dropout,
        num_dense_layers
    ):
        super().__init__()
        self.num_conv_blocks = num_conv_blocks
        self.residual_con = residual_con
        self.dense_dropout = dense_dropout
        self.num_dense_layers = num_dense_layers

        self.convnet_3_layers = {}
        self.convnet_5_layers = {}
        self.convnet_7_layers = {}

        self.res_convnet_3_layers = {}
        self.res_convnet_5_layers = {}
        self.res_convnet_7_layers = {}

        self.dense_net_layers = []

        outputs = init_neurons

        for layer_no in range(num_conv_blocks):
            for kernel in (3, 5, 7):
                if kernel == 3:
                    convnet_layers = self.convnet_3_layers
                    res_convnet_layers = self.res_convnet_3_layers
                elif kernel == 5:
                    convnet_layers = self.convnet_5_layers
                    res_convnet_layers = self.res_convnet_5_layers
                elif kernel == 7:
                    convnet_layers = self.convnet_7_layers
                    res_convnet_layers = self.res_convnet_7_layers
                else:
                    raise ValueError

                if layer_no == 0:
                    outputs = init_neurons
                else:
                    outputs = init_neurons * (layer_no * 2)

                convnet_layer = nn.Sequential(
                    self.causal_conv_1d(
                        in_channels=num_freq_bin,
                        kernel_size=kernel,
                        stride=1, out_channels=outputs
                    ),
                    nn.Linear(outputs, outputs),
                    nn.LeakyReLU(negative_slope=0.3)
                )

                print(f'CONV LAYER {layer_no} {kernel}')
                print(convnet_layer)

                if residual_con > 0 and (layer_no - residual_con) >= 0:
                    res_convnet_layer = nn.Conv1d(
                        in_channels=num_freq_bin,
                        stride=1, kernel_size=1,
                        out_channels=outputs
                    )
                else:
                    res_convnet_layer = None

                convnet_layers[layer_no] = convnet_layer
                res_convnet_layers[layer_no] = res_convnet_layer

        for layer_no in range(self.num_dense_layers):
            dense_net = nn.Sequential(
                nn.Linear(outputs, num_dense_neurons),
                nn.BatchNorm2d(
                    num_features=num_dense_neurons,
                    momentum=0.99, eps=0.001
                ),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Dropout(p=dense_dropout)
            )

            self.dense_net_layers.append(dense_net)

        self.final_dense = nn.Linear(
            num_dense_neurons, out_features=1
        )

    def forward(self, image_inputs):
        conv_values = {}
        conv_dense_layers = []

        for kernel in (3, 5, 7):
            print(f'KERNEL {kernel}')
            conv_output = image_inputs

            for layer_no in range(self.num_conv_blocks):
                print(f'LAYER NO {layer_no}')

                if kernel == 3:
                    convnet_layers = self.convnet_3_layers
                    res_convnet_layers = self.res_convnet_3_layers
                elif kernel == 5:
                    convnet_layers = self.convnet_5_layers
                    res_convnet_layers = self.res_convnet_5_layers
                elif kernel == 7:
                    convnet_layers = self.convnet_7_layers
                    res_convnet_layers = self.res_convnet_7_layers
                else:
                    raise ValueError

                convnet_layer = convnet_layers[layer_no]
                res_convnet_layer = res_convnet_layers[layer_no]
                print(f'CONVNET LAYER {convnet_layer}')
                sub_conv_output = convnet_layer(conv_output)

                if res_convnet_layer is not None:
                    res_conv_output = res_convnet_layer(conv_output)
                    conv_output = torch.add(
                        conv_output, res_conv_output
                    )
                else:
                    conv_output = sub_conv_output

            conv_dense = custom_pooling([image_inputs, conv_output])
            conv_dense_layers.append(conv_dense)
            conv_values[kernel] = conv_dense

        dense_val = torch.cat(conv_dense_layers, dim=-1)

        for layer_no in range(self.num_dense_layers):
            dense_layer = self.dense_net_layers[layer_no]
            dense_val = dense_layer(dense_val)

        final_dense_val = self.final_dense(dense_val)
        final_val = nn.sigmoid(final_dense_val)
        return final_val

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



