import os
import numpy as np
import matplotlib
import nlpaug.augmenter.audio as naa
import matplotlib.pyplot as plt
import librosa.display
import librosa.filters
import multiprocessing
import soundfile as sf

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from joblib import Parallel, delayed
from constants import model_params, base_data_path
from scipy import signal
from scipy.io import wavfile
from skopt import gp_minimize
from skopt.space import Real
from functools import partial
from pydub import AudioSegment
# from keras.utils import multi_gpu_model

from constants import *

# Set a random seed for numpy for reproducibility
np.random.seed(42)

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

try:
    import foundations
except Exception as e:
    print(e)


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr, inv_preemphasize, k):
    # wav = inv_preemphasis(wav, k, inv_preemphasize)
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


# From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    assert quantized.size > 0
    start, end = 0, quantized.size - 1

    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break

    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def trim_silence(wav, hparams):
    """
    Trim leading and trailing silence
    Useful for M-AILABS dataset if we choose to trim
    the extra 0.5 silence at beginning and end.

    Thanks @begeekmyfriend and @lautjy for pointing out
    the params contradiction. These params are separate
    and tunable per dataset.
    """
    return librosa.effects.trim(
        wav, top_db=hparams.trim_top_db,
        frame_length=hparams.trim_fft_size,
        hop_length=hparams.trim_hop_size
    )[0]


def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(
            hparams.frame_shift_ms / 1000 * hparams.sample_rate
        )
    return hop_size


def linearspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = (
        _amp_to_db(np.abs(D) ** hparams.magnitude_power, hparams) -
        hparams.ref_level_db
    )

    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(
        np.abs(D) ** hparams.magnitude_power, hparams
    ), hparams) - hparams.ref_level_db

    if hparams.signal_normalization:
        return _normalize(S, hparams)

    return S


def inv_linear_spectrogram(linear_spectrogram, hparams):
    """
    Converts linear spectrogram to waveform using librosa
    """
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    # Convert back to linear
    S = (
            _db_to_amp(D + hparams.ref_level_db) **
            (1 / hparams.magnitude_power)
    )

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(
            S.astype(np.float64).T ** hparams.power
        )
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(
            y, hparams.preemphasis, hparams.preemphasize
        )
    else:
        return inv_preemphasis(
            _griffin_lim(S ** hparams.power, hparams),
            hparams.preemphasis, hparams.preemphasize
        )


def inv_mel_spectrogram(mel_spectrogram, hparams):
    """
    Converts mel spectrogram to waveform using librosa
    """
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(
        _db_to_amp(D + hparams.ref_level_db) **
        (1 / hparams.magnitude_power),
        hparams
    )  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(
            S.astype(np.float64).T ** hparams.power
        )
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(
            y, hparams.preemphasis, hparams.preemphasize
        )
    else:
        return inv_preemphasis(
            _griffin_lim(S ** hparams.power, hparams),
            hparams.preemphasis, hparams.preemphasize
        )


# tensorflow Griffin-Lim
# Thanks to @begeekmyfriend:
# https://github.com/begeekmyfriend/Tacotron-2/blob/
# mandarin-new/datasets/audio.py

def inv_linear_spectrogram_tensorflow(spectrogram, hparams):
    """
    Builds computational graph to convert spectrogram
    to waveform using TensorFlow.
    Unlike inv_spectrogram, this does NOT invert the preemphasis.
    The caller should call
    inv_preemphasis on the output after running the graph.
    """
    if hparams.signal_normalization:
        D = _denormalize_tensorflow(spectrogram, hparams)
    else:
        D = linear_spectrogram

    S = tf.pow(
        _db_to_amp_tensorflow(D + hparams.ref_level_db),
        (1 / hparams.magnitude_power)
    )

    return _griffin_lim_tensorflow(
        tf.pow(S, hparams.power), hparams
    )


def inv_mel_spectrogram_tensorflow(mel_spectrogram, hparams):
    """
    Builds computational graph to convert mel spectrogram
    to waveform using TensorFlow.
    Unlike inv_mel_spectrogram, this does NOT invert the preemphasis.
    The caller should call
    inv_preemphasis on the output after running the graph.
    """
    if hparams.signal_normalization:
        D = _denormalize_tensorflow(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = tf.pow(
        _db_to_amp_tensorflow(D + hparams.ref_level_db),
        (1 / hparams.magnitude_power)
    )
    # Convert back to linear
    S = _mel_to_linear_tensorflow(S, hparams)
    return _griffin_lim_tensorflow(
        tf.pow(S, hparams.power), hparams
    )


def _lws_processor(hparams):
    import lws
    return lws.lws(
        hparams.n_fft, get_hop_size(hparams),
        fftsize=hparams.win_size, mode="speech"
    )

def _griffin_lim(S, hparams):
    """
    liberos implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)

    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)

    return y


def _griffin_lim_tensorflow(S, hparams):
    """
    TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises
    /blob/master/Audio_Processing.ipynb
    """

    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a
        # batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = tf.contrib.signal.inverse_stft(
            S_complex, hparams.win_size, get_hop_size(hparams),
            hparams.n_fft
        )

        for i in range(hparams.griffin_lim_iters):
            est = tf.contrib.signal.stft(
                y, hparams.win_size, get_hop_size(hparams),
                hparams.n_fft
            )
            angles = est / tf.cast(
                tf.maximum(1e-8, tf.abs(est)), tf.complex64
            )
            y = tf.contrib.signal.inverse_stft(
                S_complex * angles, hparams.win_size,
                get_hop_size(hparams), hparams.n_fft
            )

    return tf.squeeze(y, 0)


def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(
            y=y, n_fft=hparams.n_fft,
            hop_length=get_hop_size(hparams),
            win_length=hparams.win_size,
            pad_mode='constant'
        )


def _istft(y, hparams):
    return librosa.istft(
        y, hop_length=get_hop_size(hparams),
        win_length=hparams.win_size
    )


# Those are only correct when using lws!!!
# (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """
    Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """
    Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    """
    compute right padding (final frame) or both sides
    padding (first and final frames)
    """
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]

    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _mel_to_linear_tensorflow(mel_spectrogram, hparams):
    global _inv_mel_basis

    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))

    return tf.transpose(
        tf.maximum(1e-10, tf.matmul(
            tf.cast(_inv_mel_basis, tf.float32),
            tf.transpose(mel_spectrogram, [1, 0]))
        ), [1, 0]
    )


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(
        hparams.sample_rate, hparams.n_fft,
        n_mels=hparams.num_mels, fmin=hparams.fmin, fmax=hparams.fmax
    )


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * (
                (S - hparams.min_level_db) / (-hparams.min_level_db)
            ) - hparams.max_abs_value,
                -hparams.max_abs_value, hparams.max_abs_value
            )
        else:
            return np.clip(
                hparams.max_abs_value * (
                    (S - hparams.min_level_db) / -hparams.min_level_db
                ), 0, hparams.max_abs_value
            )

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * (
            (S - hparams.min_level_db) / (-hparams.min_level_db)
        ) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * (
            (S - hparams.min_level_db) / (-hparams.min_level_db)
        )


def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            clip_val = np.clip(
                D, -hparams.max_abs_value, hparams.max_abs_value
            )
            return hparams.min_level_db + (
                (clip_val + hparams.max_abs_value) *
                -hparams.min_level_db / (2 * hparams.max_abs_value)
            )
        else:
            return hparams.min_level_db + (
                np.clip(D, 0, hparams.max_abs_value) *
                -hparams.min_level_db / hparams.max_abs_value
            )

    if hparams.symmetric_mels:
        return ((
            (D + hparams.max_abs_value) *
            -hparams.min_level_db / (
                2 * hparams.max_abs_value
            )) + hparams.min_level_db
        )
    else:
        return (
            (D * -hparams.min_level_db / hparams.max_abs_value) +
            hparams.min_level_db
        )

def _denormalize_tensorflow(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return hparams.min_level_db + (
                tf.clip_by_value(
                    D, -hparams.max_abs_value, hparams.max_abs_value
                ) + hparams.max_abs_value
            ) * -hparams.min_level_db / (2 * hparams.max_abs_value)
        else:
            return (
                tf.clip_by_value(D, 0, hparams.max_abs_value) *
                -hparams.min_level_db / hparams.max_abs_value
            ) + hparams.min_level_db

    if hparams.symmetric_mels:
        return (
            (D + hparams.max_abs_value) *
            -hparams.min_level_db / (2 * hparams.max_abs_value)
        ) + hparams.min_level_db
    else:
        return (
            (D * -hparams.min_level_db / hparams.max_abs_value) +
            hparams.min_level_db
        )


# given a path, return list of all files in directory
def get_list_of_wav_files(file_path):
    files = os.listdir(file_path)
    absolute_given_dir = os.path.abspath(file_path)

    absolute_files = list(map(
        lambda path:
        os.path.join(absolute_given_dir, path), files
    ))

    return absolute_files


def convert_to_flac(dir_path):
    for file_path in os.listdir(dir_path):
        if file_path.split('.')[-1] != "flac":
            read_file = AudioSegment.from_file(
                os.path.join(dir_path, file_path),
                file_path.split('.')[-1]
            )
            os.remove(os.path.join(dir_path, file_path))
            base_name = file_path.split('.')[:-1]
            # read_file = read_file.set_channels(8)
            # base_name = ".".join(base_name)
            read_file.export(
                os.path.join(dir_path, f"{base_name[0]}.flac"),
                format="flac"
            )


def get_target(file_path):
    if '/real/' in file_path:
        return 'real'
    elif '/fake/' in file_path:
        return 'fake'


def save_wav_to_npy(output_file, spectrogram):
    np.save(output_file, spectrogram)


def wav_to_mel(input_file, output_path):
    y, sr = librosa.load(input_file)
    filename = os.path.basename(input_file)
    target = get_target(input_file)

    output_file = '{}{}-{}'.format(
        output_path, filename.split('.')[0], target
    )

    mel_spec = librosa.feature.melspectrogram
    mel_spectrogram_of_audio = mel_spec(y=y, sr=sr).T
    save_wav_to_npy(output_file, mel_spectrogram_of_audio)


def convert_and_save(
    real_audio_files, output_real, fake_audio_files, output_fake
):
    for file in real_audio_files:
        wav_to_mel(file, output_real)

    print(
        str(len(real_audio_files)) +
        ' real files converted to spectrogram'
    )

    for file in fake_audio_files:
        wav_to_mel(file, output_fake)

    print(
        str(len(fake_audio_files)) +
        ' fake files converted to spectrogram'
    )


def split_title_line(title_text, max_words=5):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([
        ' '.join(seq[i:i + max_words])
        for i in range(0, len(seq), max_words)
    ])


def plot_spectrogram(
    pred_spectrogram, path, title=None, split_title=False,
    target_spectrogram=None, max_len=None, auto_aspect=False
):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if split_title:
        title = split_title_line(title)

    fig = plt.figure(figsize=(10, 8))
    # Set common labels
    fig.text(
        0.5, 0.18, title,
        horizontalalignment='center', fontsize=16
    )

    # target spectrogram subplot
    if target_spectrogram is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)

        if auto_aspect:
            im = ax1.imshow(
                np.rot90(target_spectrogram), aspect='auto',
                interpolation='none'
            )
        else:
            im = ax1.imshow(
                np.rot90(target_spectrogram),
                interpolation='none'
            )

        ax1.set_title('Target Mel-Spectrogram')
        fig.colorbar(
            mappable=im, shrink=0.65,
            orientation='horizontal', ax=ax1
        )

        ax2.set_title('Predicted Mel-Spectrogram')
    else:
        ax2 = fig.add_subplot(211)

    if auto_aspect:
        im = ax2.imshow(
            np.rot90(pred_spectrogram), aspect='auto',
            interpolation='none'
        )
    else:
        im = ax2.imshow(
            np.rot90(pred_spectrogram),
            interpolation='none'
        )

    fig.colorbar(
        mappable=im, shrink=0.65,
        orientation='horizontal', ax=ax2
    )

    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


def process_audio_files(filename, dirpath):
    audio_array, sample_rate = librosa.load(
        os.path.join(dirpath, 'flac', filename), sr=16000
    )
    trim_audio_array, index = librosa.effects.trim(audio_array)
    mel_spec_array = melspectrogram(
        trim_audio_array, hparams=hparams
    ).T

    """
    mel_spec_array = librosa.feature.melspectrogram(
        y=trim_audio_array, sr=sample_rate, 
        n_mels=model_params['num_freq_bin'
    ]).T
    """

    label_name = filename.split('_')[-1].split('.')[0]
    if (label_name == 'bonafide') or ('target' in label_name):
        label = 1
    elif label_name == 'spoof':
        label = 0
    else:
        label = None
    if label is None:
        print(f"Removing {filename} since it does not have label")
        os.remove(os.path.join(dirpath, 'flac', filename))

    return mel_spec_array, label

def convert_audio_to_processed_list(
    input_audio_array_list, filename, dirpath
):
    label_name = filename.split('_')[-1].split('.')[0]
    out_list = []

    if label_name == 'spoof':
        audio_array_list = [input_audio_array_list[0]]
        choose_random_one_ind = np.random.choice(
            np.arange(1, len(input_audio_array_list))
        )
        audio_array_list.append(
            input_audio_array_list[choose_random_one_ind]
        )
        label = 0

    elif (label_name == 'bonafide') or ('target' in label_name):
        audio_array_list = input_audio_array_list
        label = 1
    else:
        audio_array_list = [input_audio_array_list[0]]
        label = None

    for audio_array in audio_array_list:
        trim_audio_array, index = librosa.effects.trim(audio_array)
        mel_spec_array = melspectrogram(
            trim_audio_array, hparams=hparams
        ).T

        """
        mel_spec_array = librosa.feature.melspectrogram(
            y=trim_audio_array, sr=sample_rate, 
            n_mels=model_params['num_freq_bin']
        ).T
        """

        if label is None:
            print(f"Removing {filename} since it does not have label")
            os.remove(os.path.join(dirpath, 'flac', filename))

        out_list.append([mel_spec_array, label])

    return out_list


def preprocess_and_save_audio_from_ray_parallel(
    dirpath, mode, recompute=False, dir_num=None, isaug=False
):
    if isaug:
        preproc_filename = f'{mode}_preproc_aug.npy'
    else:
        preproc_filename = f'{mode}_preproc.npy'

    # if mode != 'train':
    #     preproc_filename = f'{mode}_preproc.npy'

    if dir_num is not None:
        base_path = base_data_path[dir_num]
    else:
        base_path = base_data_path[0]

    is_file = os.path.isfile(os.path.join(
        f'{base_path}/preprocessed_data', preproc_filename
    ))

    if not is_file or recompute:
        filenames = os.listdir(os.path.join(dirpath, 'flac'))
        num_cores = multiprocessing.cpu_count() - 1

        if isaug:
            precproc_list_saved = Parallel(n_jobs=num_cores)(
                delayed(process_audio_files_with_aug)(
                    filename, dirpath
                ) for filename in tqdm(filenames)
            )

            # Flatten the list
            print(
                f"******original len of preproc_list:",
                len(precproc_list_saved)
            )
            precproc_list = []
            for i in range(len(precproc_list_saved)):
                precproc_list.extend(precproc_list_saved[i])

            """
            precproc_list = [
                item for sublist in precproc_list
                 for item in sublist
            ]
            """
            print(
                f"******flattened len of preproc_list:",
                len(precproc_list)
            )
        else:
            precproc_list = Parallel(n_jobs=num_cores)(
                delayed(process_audio_files)(filename, dirpath)
                for filename in tqdm(filenames)
            )

        precproc_list = [x for x in precproc_list if x[1] is not None]

        if not os.path.isdir(f'{base_path}/preprocessed_data'):
            os.mkdir(f'{base_path}/preprocessed_data')

        np.save(os.path.join(
            f'{base_path}/preprocessed_data', preproc_filename
        ), precproc_list)
    else:
        print("Preprocessing already done!")

def process(*args, **kwargs):
    return process_audio_files_inference(*args, **kwargs)

def process_audio_files_inference(
    filename, dirpath, mode, normalize=False
):
    if type(filename) is tuple:
        filename = os.path.join(*filename)
    elif type(filename) == np.ndarray:
        filename = os.path.join(*filename)

    path = os.path.join(dirpath, filename)

    audio_array, sample_rate = librosa.load(path, sr=16000)
    if normalize:
        rms = np.sqrt(np.mean(audio_array ** 2))
        audio_array /= rms

    trim_audio_array, index = librosa.effects.trim(audio_array)
    mel_spec_array = melspectrogram(
        trim_audio_array, hparams=hparams
    ).T

    # https://stackoverflow.com/questions/57072513/
    duration = get_duration(filename)

    if mode == 'unlabeled':
        return mel_spec_array
    elif mode == 'real':
        label = 0
    elif mode == 'fake':
        label = 1
    elif mode in (0, 1):
        label = mode
    else:
        raise ValueError(f'BAD MODE {mode}')

    return mel_spec_array, label, duration


def get_durations(filenames, dirpath='', show_pbar=True):
    if show_pbar:
        iterable = tqdm(range(len(filenames)))
    else:
        iterable = range(len(filenames))

    durations = []
    for k in iterable:
        filename = filenames[k]
        if show_pbar:
            iterable.set_description(str(filename))

        duration = get_duration(filename, dirpath)
        durations.append(duration)

    return durations


def get_duration(filename, dirpath=''):
    if type(filename) is tuple:
        filename = os.path.join(*filename)

    file_path = os.path.join(dirpath, filename)
    file = sf.SoundFile(file_path)
    duration = file.frames / file.samplerate
    return duration

def get_frames(filename, dirpath=''):
    if type(filename) is tuple:
        filename = os.path.join(*filename)

    file_path = os.path.join(dirpath, filename)
    file = sf.SoundFile(file_path)
    frames = file.frames
    return frames


def preprocess_from_filenames(
    filenames, dirpath, mode, use_parallel=True,
    show_pbar=True, num_cores=None, func=process,
    cache=None, cache_threshold=30, normalize=False
):
    if show_pbar:
        iterable = tqdm(range(len(filenames)))
    else:
        iterable = range(len(filenames))

    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    arg_list = []
    cache_list = []

    if use_parallel:
        process_list = []

        for k in iterable:
            filename = filenames[k]
            if type(filename) is tuple:
                filename = os.path.join(*filename)

            if type(mode) is dict:
                file_mode = mode[filename]
            elif type(mode) in (list, tuple):
                assert len(mode) == len(filenames)
                file_mode = mode[k]
            else:
                file_mode = mode

            delayed_func = delayed(func)
            args = (filename, dirpath, file_mode, normalize)

            if args in cache:
                data = cache[args]
                cache_list.append(data)
                continue

            process = delayed_func(*args)
            process_list.append(process)
            arg_list.append(args)

        preproc_list = Parallel(n_jobs=num_cores)(process_list)

    else:
        preproc_list = []
        for k in iterable:
            filename = filenames[k]

            if type(mode) is dict:
                file_mode = mode[filename]
            elif type(mode) in (list, tuple):
                assert len(mode) == len(filenames)
                file_mode = mode[k]
            else:
                file_mode = mode

            args = (filename, dirpath, file_mode, normalize)
            if args in cache:
                data = self.cache[args]
                preproc_list.append(data)
                continue

            preproc_list.append(func(*args))
            arg_list.append(args)

    durations = []
    for k, data in enumerate(preproc_list):
        mel_spec_array, label, duration = data
        durations.append(duration)
        args = arg_list[k]

        if (duration > cache_threshold) and (args not in cache):
            cache[args] = data

    # print('MAX DURATIONS', max(durations))
    preproc_list.extend(cache_list)
    return preproc_list


def preprocess_parallel(*args, **kwargs):
    return preprocess_from_ray_parallel_inference(*args, **kwargs)

def preprocess_from_ray_parallel_inference(
    dirpath, mode, use_parallel=True
):
    filenames = os.listdir(os.path.join(dirpath, mode))
    return preprocess_from_filenames(
        filenames=filenames, dirpath=dirpath, mode=mode,
        use_parallel=use_parallel
    )


def preprocess_and_save_audio_from_ray(dirpath, mode, recompute=False):
    filenames = os.listdir(os.path.join(dirpath, 'flac'))
    is_file = os.path.isfile(os.path.join(
        f'{base_data_path}/preprocessed_data', f'{mode}_preproc.npy'
    ))

    if not is_file or recompute:
        precproc_list = []

        for filename in tqdm(filenames):
            audio_array, sample_rate = librosa.load(os.path.join(
                dirpath, 'flac', filename
            ), sr=16000)

            trim_audio_array, index = librosa.effects.trim(audio_array)
            mel_spec_array = melspectrogram(
                trim_audio_array, hparams=hparams
            ).T

            """
            mel_spec_array = librosa.feature.melspectrogram(
                y=trim_audio_array, sr=sample_rate, 
                n_mels=model_params['num_freq_bin']
            ).T
            """
            label_name = filename.split('_')[-1].split('.')[0]
            if label_name == 'bonafide':
                label = 1
            elif label_name == 'spoof':
                label = 0
            else:
                label = None
            if label is not None:
                precproc_list.append((mel_spec_array, label))
            if label is None:
                print(
                    f"Removing {filename} since it does not have label"
                )
                os.remove(os.path.join(dirpath, 'flac', filename))

        if not os.path.isdir(f'{base_data_path}/preprocessed_data'):
            os.mkdir(f'{base_data_path}/preprocessed_data')

        np.save(os.path.join(
            f'{base_data_path}/preprocessed_data', f'{mode}_preproc.npy'
        ), precproc_list)

        """
        np.save(os.path.join(
            dirpath, 'preproc', 'preproc.npy'
        ), precproc_list)
        """
    else:
        print("Preprocessing already done!")


def preprocess_and_save_audio(dirpath, recompute=False):
    filenames = os.listdir(os.path.join(dirpath, 'flac'))
    is_file = os.path.isfile(os.path.join(
        dirpath, 'preproc', 'preproc.npy'
    ))

    if not is_file or recompute:
        precproc_list = []

        for filename in tqdm(filenames):
            audio_array, sample_rate = librosa.load(os.path.join(
                dirpath, 'flac', filename
            ), sr=16000)

            trim_audio_array, index = librosa.effects.trim(audio_array)
            mel_spec_array = librosa.feature.melspectrogram(
                y=trim_audio_array, sr=sample_rate,
                n_mels=model_params['num_freq_bin']
            ).T

            label_name = filename.split('_')[-1].split('.')[0]

            if label_name == 'bonafide':
                label = 1
            elif label_name == 'spoof':
                label = 0
            else:
                label = None

            if label is not None:
                precproc_list.append((mel_spec_array, label))
            if label is None:
                print(
                    f"Removing {filename} since it does not have label"
                )
                os.remove(os.path.join(dirpath, 'flac', filename))

        if not os.path.isdir(os.path.join(dirpath, 'preproc')):
            os.mkdir(os.path.join(dirpath, 'preproc'))

        np.save(os.path.join(
            dirpath, 'preproc', 'preproc.npy'
        ), precproc_list)
    else:
        print("Preprocessing already done!")


def describe_array(arr):
    print(
        f"Mean duration: {arr.mean()}" +
        "\nStandard Deviation: {arr.std()}" +
        "\nNumber of Clips: {len(arr)}"
    )
    plt.hist(arr, bins=40)
    plt.show()


def get_durations_from_dir(audio_dir, file_extension='.wav'):
    durations = list()

    for root, dirs, filenames in os.walk(audio_dir):
        for file_name in filenames:
            if file_extension in file_name:
                file_path = os.path.join(root, file_name)
                audio = AudioSegment.from_wav(file_path)
                duration = audio.duration_seconds
                durations.append(duration)

    return np.array(durations)


def get_zero_pad(batch_input):
    # find max length
    max_length = np.max([len(x) for x in batch_input])

    for i, arr in enumerate(batch_input):
        curr_length = len(arr)
        pad_length = max_length - curr_length

        if len(arr.shape) > 1:
            arr = np.concatenate([
                arr, np.zeros((pad_length, arr.shape[-1]))
            ])
        else:
            arr = np.concatenate([arr, np.zeros(pad_length)])

        batch_input[i] = arr

    return batch_input


def truncate_array(batch_input):
    min_arr_len = np.min([len(x) for x in batch_input])
    for i, arr in enumerate(batch_input):
        batch_input[i] = arr[:min_arr_len]
    return batch_input


def random_truncate_array(batch_input):
    min_arr_len = np.min([len(x) for x in batch_input])

    for i, arr in enumerate(batch_input):
        upper_limit_start_point = len(arr) - min_arr_len

        if upper_limit_start_point > 0:
            start_point = np.random.randint(0, upper_limit_start_point)
        else:
            start_point = 0

        batch_input[i] = arr[start_point:(start_point + min_arr_len)]

    return batch_input


class f1_score_callback(object):
    def __init__(
        self, x_val_inp, y_val_inp, model_save_filename=None,
        save_model=True
    ):
        self.x_val = x_val_inp
        self.y_val = y_val_inp
        self.model_save_filename = model_save_filename
        self.save_model = save_model
        self._val_f1 = 0

        self.f1_score_value = None

    def on_train_begin(self, logs=None):
        self.f1_score_value = []

    def on_epoch_end(self, epoch, logs=None):
        y_val = self.y_val
        datagen_val = DataGenerator(self.x_val, mode='test')
        y_pred = self.model.predict_generator(
            datagen_val, use_multiprocessing=False, max_queue_size=50
        )
        y_pred_labels = np.zeros((len(y_pred)))
        y_pred_labels[y_pred.flatten() > 0.5] = 1

        self._val_f1 = f1_score(y_val, y_pred_labels.astype(int))
        print(f"val_f1: {self._val_f1:.4f}")
        self.f1_score_value.append(self._val_f1)

        if self.save_model:
            if self._val_f1 >= max(self.f1_score_value):
                print("F1 score has improved. Saving model.")
                self.model.save(self.model_save_filename)

        try:
            foundations.log_metric('epoch_val_f1_score', self._val_f1)
            foundations.log_metric(
                'best_f1_score', max(self.f1_score_value)
            )
        except Exception as e:
            print(e)

        return


class DataGenerator(object):
    def __init__(
        self, x_set, y_set=None, sample_weights=None,
        batch_size=model_params['batch_size'], shuffle=False,
        mode='train'
    ):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.sample_weights = sample_weights

        if self.mode != 'train':
            self.shuffle = False

        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_x = get_zero_pad(batch_x)
        # batch_x = random_truncate_array(batch_x)
        batch_x = np.array(batch_x)
        batch_x = batch_x.reshape((len(batch_x), -1, hparams.num_mels))

        if self.mode != 'test':
            batch_y = self.y[
                idx * self.batch_size:(idx + 1) * self.batch_size
            ]

            # read your data here using the batch lists,
            # batch_x and batch_y

            if self.mode == 'train':
                return np.array(batch_x), np.array(batch_y)
            if self.mode == 'val':
                return np.array(batch_x), np.array(batch_y)
        else:
            return np.array(batch_x)

    def __next__(self):
        if self.n >= self.max:
            self.n = 0

        result = self.__getitem__(self.n)
        self.n += 1
        return result
