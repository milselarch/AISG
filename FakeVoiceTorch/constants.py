import os
import numpy as np
import yaml

config_yaml_file_name = 'config.yml'

class HParams(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, item):
        kwargs = super().__getattribute__('kwargs')
        return kwargs[item]

    def __getitem__(self, item):
        kwargs = super().__getattribute__('kwargs')
        return kwargs[item]


hparams = HParams(
    cleaners='english_cleaners',

    tacotron_num_gpus=1,
    wavenet_num_gpus=1,
    split_on_cpu=True,

    num_mels=240,
    num_freq=1025,
    rescale=True,
    rescaling_max=0.999,

    clip_mels_length=False,
    max_mel_frames=900,

    use_lws=False,
    silence_threshold=2,

    n_fft=1000,
    hop_size=200,
    win_size=800,
    sample_rate=16000,
    frame_shift_ms=None,
    magnitude_power=2.,

    trim_silence=True,
    trim_fft_size=2048,
    trim_hop_size=512,
    trim_top_db=40,

    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=1.,

    normalize_for_wavenet=True,
    clip_for_wavenet=True,

    wavenet_pad_sides=1,

    preemphasize=True,
    preemphasis=0.97,

    min_level_db=-100,
    ref_level_db=20,
    fmin=0,
    fmax=8000,

    power=1.5,
    griffin_lim_iters=60,
    GL_on_GPU=True,

    outputs_per_step=1,
    stop_at_any=True,
    batch_norm_position='after',
    clip_outputs=True,
    lower_bound_decay=0.1,

    embedding_dim=512,

    enc_conv_num_layers=3,
    enc_conv_kernel_size=(5,),
    enc_conv_channels=512,
    encoder_lstm_units=256,

    smoothing=False,
    attention_dim=128,
    attention_filters=32,
    attention_kernel=(31,),
    cumulative_weights=True,

    synthesis_constraint=False,
    synthesis_constraint_type='window',
    attention_win_size=7,

    prenet_layers=[256, 256],
    decoder_layers=2,
    decoder_lstm_units=1024,
    max_iters=10000,

    postnet_num_layers=5,
    postnet_kernel_size=(5,),
    postnet_channels=512,

    cbhg_kernels=8,
    cbhg_conv_channels=128,
    cbhg_pool_size=2,
    cbhg_projection=256,
    cbhg_projection_kernel_size=3,
    cbhg_highwaynet_layers=4,
    cbhg_highway_units=128,
    cbhg_rnn_units=128,

    mask_encoder=True,
    mask_decoder=False,
    cross_entropy_pos_weight=1,
    predict_linear=True,

    input_type="raw",
    quantize_channels=2 ** 16,
    use_bias=True,
    legacy=True,
    residual_legacy=True,

    log_scale_min=float(np.log(1e-14)),
    log_scale_min_gauss=float(np.log(1e-7)),

    cdf_loss=False,

    out_channels=2,
    layers=20,
    stacks=2,
    residual_channels=128,
    gate_channels=256,
    skip_out_channels=128,
    kernel_size=3,

    cin_channels=80,

    upsample_type='SubPixel',
    upsample_activation='Relu',
    upsample_scales=[11, 25],
    freq_axis_kernel_size=3,
    leaky_alpha=0.4,
    NN_init=True,
    NN_scaler=0.3,

    gin_channels=-1,
    use_speaker_embedding=True,
    n_speakers=5,
    speakers_path=None,
    speakers=['speaker0', 'speaker1',
              'speaker2', 'speaker3', 'speaker4'],

    tacotron_random_seed=5339,
    tacotron_data_random_state=1234,

    tacotron_swap_with_cpu=False,

    tacotron_batch_size=32,

    tacotron_synthesis_batch_size=1,
    tacotron_test_size=0.05,
    tacotron_test_batches=None,

    tacotron_decay_learning_rate=True,
    tacotron_start_decay=40000,
    tacotron_decay_steps=18000,
    tacotron_decay_rate=0.5,
    tacotron_initial_learning_rate=1e-3,
    tacotron_final_learning_rate=1e-4,

    tacotron_adam_beta1=0.9,
    tacotron_adam_beta2=0.999,
    tacotron_adam_epsilon=1e-6,

    tacotron_reg_weight=1e-6,
    tacotron_scale_regularization=False,
    tacotron_zoneout_rate=0.1,
    tacotron_dropout_rate=0.5,
    tacotron_clip_gradients=True,

    tacotron_natural_eval=False,

    tacotron_teacher_forcing_mode='constant',
    tacotron_teacher_forcing_ratio=1.,
    tacotron_teacher_forcing_init_ratio=1.,
    tacotron_teacher_forcing_final_ratio=0.,
    tacotron_teacher_forcing_start_decay=10000,
    tacotron_teacher_forcing_decay_steps=40000,
    tacotron_teacher_forcing_decay_alpha=None,

    tacotron_fine_tuning=False,

    wavenet_random_seed=5339,
    wavenet_data_random_state=1234,

    wavenet_swap_with_cpu=False,

    wavenet_batch_size=8,

    wavenet_synthesis_batch_size=10 * 2,
    wavenet_test_size=None,
    wavenet_test_batches=1,

    wavenet_lr_schedule='exponential',
    wavenet_learning_rate=1e-3,
    wavenet_warmup=float(4000),
    wavenet_decay_rate=0.5,
    wavenet_decay_steps=200000,

    wavenet_adam_beta1=0.9,
    wavenet_adam_beta2=0.999,
    wavenet_adam_epsilon=1e-6,

    wavenet_clip_gradients=True,
    wavenet_ema_decay=0.9999,
    wavenet_weight_normalization=False,
    wavenet_init_scale=1.,
    wavenet_dropout=0.05,
    wavenet_gradient_max_norm=100.0,
    wavenet_gradient_max_value=5.0,

    max_time_sec=None,
    max_time_steps=11000,

    wavenet_natural_eval=False,

    train_with_GTA=True,

    sentences=[
        'Scientists at the CERN laboratory say they have discovered a new particle.',
        'There\'s a way to measure the acute emotional intelligence that has never gone out of style.',
        'President Trump met with other leaders at the Group of 20 conference.',
        'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',

        'Generative adversarial network or variational auto-encoder.',
        'Basilar membrane and otolaryngology are not auto-correlations.',
        'He has read the whole thing.',
        'He reads books.',
        'He thought it was time to present the present.',
        'Thisss isrealy awhsome.',
        'The big brown fox jumps over the lazy dog.',
        'Did the big brown fox jump over the lazy dog?',
        "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
        "She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
        "Tajima Airport serves Toyooka.",

        'Thank you so much for your support!',
    ],

    wavenet_synth_debug=False,
    wavenet_debug_wavs=['training_data/audio/audio-LJ001-0008.npy'],
    wavenet_debug_mels=['training_data/mels/mel-LJ001-0008.npy'],
)


def random_select_from_list(valid_value_list):
    random_ind = np.random.randint(len(valid_value_list))
    return valid_value_list[random_ind]


def generate_config(config_yaml='config.yml'):
    config_dict = {
        'num_freq_bin': [hparams.num_mels],
        'num_conv_blocks': [4, 8, 12, 16],
        'num_conv_filters': [16, 32, 64],
        'spatial_dropout_fraction': [0, 0.05, 0.1],
        'num_dense_layers': [1, 2, 3, 4, 5],
        'num_dense_neurons': [10, 50, 100, 150, 200],
        'dense_dropout': [0, 0.05, 0.1],
        'learning_rate': [0.001, 0.0001],
        'epochs': [200, 500, 1000],
        'batch_size': [64, 156, 256],
        'residual_con': [0, 2, 4],
        'use_default': [False],
        'model_save_dir': ['fitted_objects']
    }

    for k, v in config_dict.items():
        config_dict[k] = random_select_from_list(v)

    with open(config_yaml_file_name, 'w') as outfile:
        yaml.dump(config_yaml, outfile, default_flow_style=False)


def load_config_yaml():
    with open(config_yaml_file_name, 'r') as yfile:
        yaml_config_dict = yaml.safe_load(yfile)
    return yaml_config_dict


model_params = load_config_yaml()

if model_params['use_default']:
    model_params = {
        'num_freq_bin': hparams.num_mels,
        'num_conv_blocks': 8,
        'num_conv_filters': 32,
        'spatial_dropout_fraction': 0.05,
        'num_dense_layers': 1,
        'num_dense_neurons': 50,
        'dense_dropout': 0,
        'learning_rate': 0.0001,
        'epochs': 1,
        'batch_size': 156,
        'residual_con': 2,
        'use_default': True,
        'model_save_dir': 'fitted_objects'
    }

run_on_foundations = True

if run_on_foundations:
    base_data_path = ['/data/logical_access']
else:
    base_data_path = ['../data/logical_access']

measure_performance_only = False
