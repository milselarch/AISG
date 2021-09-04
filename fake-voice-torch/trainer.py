import os
import model

from constants import model_params, base_data_path

class Trainer(object):
    def __init__(self):
        self.model = self.make_model()

    def make_model(self):
        discriminator = model.Discriminator(
            num_freq_bin=model_params['num_freq_bin'],
            init_neurons=model_params['num_conv_filters'],
            num_conv_blocks=model_params['num_conv_blocks'],
            residual_con=model_params['residual_con'],
            num_dense_neurons=model_params['num_dense_neurons'],
            dense_dropout=model_params['dense_dropout'],
            num_dense_layers=model_params['num_dense_layers'],
        )

        return discriminator