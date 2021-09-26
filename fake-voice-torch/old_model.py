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
        num_dense_layers, spatial_dropout_fraction=0
    ):
        super().__init__()
        self.num_conv_blocks = num_conv_blocks
        self.residual_con = residual_con
        self.dense_dropout = dense_dropout
        self.num_dense_layers = num_dense_layers
        self.spatial_dropout_fraction = spatial_dropout_fraction

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

        self.nn_convnet_3 = self.convert(self.convnet_3_layers)
        self.nn_convnet_5 = self.convert(self.convnet_5_layers)
        self.nn_convnet_7 = self.convert(self.convnet_7_layers)

        self.nn_r_convnet_3 = self.convert(self.res_convnet_3_layers)
        self.nn_r_convnet_5 = self.convert(self.res_convnet_5_layers)
        self.nn_r_convnet_7 = self.convert(self.res_convnet_7_layers)

        self.nn_dense_layers = nn.ModuleList(self.dense_net_layers)

        self.final_dense = mark(nn.Linear(
            num_dense_neurons, out_features=1
        ))

    @staticmethod
    def convert(mapping):
        new_mapping = {}
        for k in mapping:
            new_mapping[str(k)] = mapping[k]

        return nn.ModuleDict(new_mapping)

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

    def forward_image(self, image_inputs):
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

        shapes = [ly.shape for ly in conv_dense_layers]
        # print(f'PRE-CAT SHAPES {shapes}')
        dense_val = torch.cat(conv_dense_layers, dim=-1)
        # print(f'DENSE {dense_val.shape}')

        for layer_no in range(self.num_dense_layers):
            dense_layer = self.dense_net_layers[layer_no]
            dense_val = dense_layer(dense_val)
            # print(f'DENSE VAL {layer_no} {dense_val.shape}')

        final_dense_val = self.final_dense(dense_val)
        final_val = torch.sigmoid(final_dense_val)
        return final_val

    def forward(self, image_inputs):
        shape_size = len(image_inputs.shape)
        if shape_size == 3:
            return self.forward_image(image_inputs)

        assert shape_size == 4

        batch_outputs = []
        for image_input in image_inputs:
            episode = self.forward_image(image_input)
            episode = torch.unsqueeze(episode, 0)
            batch_outputs.append(episode)

        batch_outputs = torch.cat(batch_outputs, dim=0)
        return batch_outputs

    @classmethod
    def spatial_dropout_1d(cls, tensor, prob):
        # https://stackoverflow.com/questions/50393666/
        # how-to-understand-spatialdropout1d-and-when-to-use-it
        # https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2
        tensor = tensor.permute(0, 2, 1)
        tensor = nn.functional.dropout2d(tensor, prob)
        tensor = tensor.permute(0, 2, 1)
        return tensor

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


if __name__ == '__main__':
    from constants import model_params

    discriminator = Discriminator(
        num_freq_bin=model_params['num_freq_bin'],
        init_neurons=model_params['num_conv_filters'],
        num_conv_blocks=model_params['num_conv_blocks'],
        residual_con=model_params['residual_con'],
        num_dense_neurons=model_params['num_dense_neurons'],
        dense_dropout=model_params['dense_dropout'],
        num_dense_layers=model_params['num_dense_layers'],
        spatial_dropout_fraction=model_params[
            'spatial_dropout_fraction'
        ]
    )

