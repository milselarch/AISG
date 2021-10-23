import torch
import torch.nn as nn

def custom_pooling(x):
    target = x[1]
    inputs = x[0]
    mask_value = 0

    # getting the mask by observing the model's inputs
    mask = torch.eq(inputs, mask_value)
    # mask = K.all(mask, axis=-1, keepdims=True)
    mask = torch.all(mask, dim=-2, keepdim=True)

    # inverting the mask for getting the valid steps for each sample
    mask = 1 - mask.float()

    # summing the valid steps for each sample
    steps_per_sample = torch.sum(mask, dim=2, keepdim=False)

    # applying the mask to the target
    # (to make sure you are summing zeros below)
    target = target * mask

    # calculating the mean of the steps
    # (using our sum of valid steps as averager)
    total = torch.sum(target, dim=2, keepdim=False)
    means = total / steps_per_sample
    return means

class Discriminator(nn.Module):
    def __init__(
        self, num_freq_bin, init_neurons, num_conv_blocks,
        residual_con, num_dense_neurons, dense_dropout,
        spatial_dropout_fraction,
        num_dense_layers, hidden_size=None, rnn_layers=1,
        final_only=False
    ):
        if hidden_size is None:
            hidden_size = num_dense_neurons

        super().__init__()
        self.num_conv_blocks = num_conv_blocks
        self.residual_con = residual_con
        self.dense_dropout = dense_dropout
        self.num_dense_layers = num_dense_layers
        self.spatial_dropout_fraction = spatial_dropout_fraction
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

        self.final_only = final_only

        self.markers = []
        self.convnet_3_layers = {}
        self.convnet_5_layers = {}
        self.convnet_7_layers = {}

        self.res_convnet_3_layers = {}
        self.res_convnet_5_layers = {}
        self.res_convnet_7_layers = {}

        self.dense_net_layers = []
        self.kernels = (3, 5, 7)
        self.neg_slope = 0.01

        mark = self.mark
        prev_outputs = num_freq_bin

        for kernel in self.kernels:
            prev_outputs = num_freq_bin

            for layer_no in range(num_conv_blocks):
                if layer_no == 0:
                    outputs = init_neurons
                else:
                    outputs = init_neurons * (layer_no * 2)

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

                convnet_layer = mark(nn.Sequential(
                    self.causal_conv_1d(
                        in_channels=prev_outputs,
                        kernel_size=kernel,
                        stride=1, out_channels=outputs
                    ),
                    nn.Linear(kernel, 1),
                    nn.LeakyReLU(negative_slope=self.neg_slope)
                ))

                convnet_layers[layer_no] = convnet_layer
                # print(f'CONV LAYER {layer_no} {kernel}')
                # print(convnet_layer)

                if residual_con > 0 and (layer_no - residual_con) >= 0:
                    res_convnet_layer = mark(nn.Sequential(
                        nn.Conv1d(
                            in_channels=prev_outputs, stride=1,
                            kernel_size=1, out_channels=outputs
                        ),
                        nn.Linear(1, 1)
                    ))
                    # print(f'RES CONV LAYER {layer_no} {kernel}')
                    # print(res_convnet_layer)
                else:
                    res_convnet_layer = None

                res_convnet_layers[layer_no] = res_convnet_layer
                prev_outputs = outputs

        for layer_no in range(self.num_dense_layers):
            if layer_no == 0:
                input_neurons = prev_outputs * len(self.kernels)
            else:
                input_neurons = num_dense_neurons

            dense_net = mark(nn.Sequential(
                nn.Linear(input_neurons, num_dense_neurons),
                nn.BatchNorm1d(
                    num_features=num_dense_neurons,
                    momentum=0.99, eps=0.001
                ),
                nn.LeakyReLU(negative_slope=self.neg_slope),
                nn.Dropout(p=dense_dropout)
            ))

            self.dense_net_layers.append(dense_net)

        self.rnn = mark(nn.LSTM(
            num_dense_neurons, self.hidden_size,
            self.rnn_layers, batch_first=True
        ))

        self.final_dense = mark(nn.Linear(
            self.hidden_size, out_features=1
        ))

    @classmethod
    def spatial_dropout_1d(cls, tensor, prob):
        # https://stackoverflow.com/questions/50393666/
        # how-to-understand-spatialdropout1d-and-when-to-use-it
        # https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
        tensor = tensor.permute(0, 2, 1)
        tensor = nn.functional.dropout2d(tensor, prob)
        tensor = tensor.permute(0, 2, 1)
        return tensor

    def to_cuda(self):
        super().cuda()
        for tensor in self.markers:
            # assert isinstance(tensor, torch.Tensor)
            tensor.cuda()

    def mark(self, tensor):
        # assert isinstance(tensor, torch.Tensor)
        self.markers.append(tensor)
        return tensor

    def load_parameters(self):
        parameters = []

        for layer in self.markers:
            sub_params = layer.parameters()
            sub_params = list(sub_params)
            parameters.extend(sub_params)

        parameters = tuple(parameters)
        return parameters

    def test(self, image_inputs, kernel=3):
        conv_output = image_inputs

        for layer_no in range(self.num_conv_blocks):
            # print(f'LAYER NO {layer_no}')

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
            # print(f'CONVNET LAYER {convnet_layer}')
            sub_conv_output = convnet_layer(conv_output)
            # print(f'CONV SHAPE {sub_conv_output.shape}')

            if res_convnet_layer is not None:
                # print(f'RES CONVNET LAYER {res_convnet_layer}')
                res_conv_output = res_convnet_layer(conv_output)
                # print(f'RES CONV SHAPE {res_conv_output.shape}')
                conv_output = sub_conv_output.add(res_conv_output)
                # print(f'CONV ADD SHAPE {conv_output.shape}')
            else:
                conv_output = sub_conv_output

        # print('POOL SHAPES')
        # print(image_inputs.shape, conv_output.shape)
        return conv_output

    def conv_series(self, image_inputs):
        conv_values = {}
        conv_dense_layers = []

        for kernel in (3, 5, 7):
            # print(f'KERNEL {kernel}')
            conv_output = image_inputs

            for layer_no in range(self.num_conv_blocks):
                # print(f'LAYER NO {layer_no}')

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
                # print(f'CONVNET LAYER {convnet_layer}')
                sub_conv_output = convnet_layer(conv_output)
                # print(f'CONV SHAPE {sub_conv_output.shape}')

                if res_convnet_layer is not None:
                    # print(f'RES CONVNET LAYER {res_convnet_layer}')
                    res_conv_output = res_convnet_layer(conv_output)
                    # print(f'RES CONV SHAPE {res_conv_output.shape}')
                    conv_output = sub_conv_output.add(res_conv_output)
                    # print(f'CONV ADD SHAPE {conv_output.shape}')
                else:
                    conv_output = sub_conv_output

                if layer_no < self.num_conv_blocks - 1:
                    conv_output = self.spatial_dropout_1d(
                        conv_output, prob=self.spatial_dropout_fraction
                    )

            # print('POOL SHAPES')
            # print(image_inputs.shape, conv_output.shape)
            conv_dense = custom_pooling([image_inputs, conv_output])
            # print('CONV DENSE', conv_dense.shape)
            conv_dense_layers.append(conv_dense)
            conv_values[kernel] = conv_dense

        dense_val = torch.cat(conv_dense_layers, dim=-1)

        for layer_no in range(self.num_dense_layers):
            dense_layer = self.dense_net_layers[layer_no]
            dense_val = dense_layer(dense_val)
            # print(f'DENSE VAL {layer_no} {dense_val.shape}')

        return dense_val

    def rnn_test(self, batch_image_inputs):
        batch_outputs = []

        for image_inputs in batch_image_inputs:
            outputs = self.conv_series(image_inputs)
            batch_outputs.append(outputs.unsqueeze(0))

        torch_batch_outputs = torch.vstack(batch_outputs)

        shapes = [t.shape for t in batch_outputs]
        print(f'SHAPES {shapes}')
        print(f'BATCH-O {torch_batch_outputs.shape}')

        rnn_output, hidden_states = self.rnn(torch_batch_outputs)
        return rnn_output, hidden_states

    def forward(self, batch_image_inputs, sigmoid=True):
        batch_outputs = []

        for image_inputs in batch_image_inputs:
            outputs = self.conv_series(image_inputs)
            batch_outputs.append(outputs.unsqueeze(0))

        torch_batch_outputs = torch.vstack(batch_outputs)
        rnn_output, hidden_states = self.rnn(torch_batch_outputs)
        # print('RNN OUTPUT', rnn_output)

        if self.final_only:
            rnn_output = rnn_output[:, -1, :]

        final_val = self.final_dense(rnn_output)

        if sigmoid:
            final_val = torch.sigmoid(final_val)

        return final_val, hidden_states

    @classmethod
    def causal_conv_1d(
        cls, in_channels, out_channels, kernel_size,
        dilation=1, **kwargs
    ):
        pad = (kernel_size - 1) * dilation

        return nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=pad, dilation=dilation,
            **kwargs
        )
