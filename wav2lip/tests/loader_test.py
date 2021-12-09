from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 1
syncnet_mel_step_size = 16

model = SyncNet(syncnet_T)

vid_name = '0ae95b34e9481b4f'
vid_dir = f'lrs2_preprocessed/videos/{vid_name}'
start_id = 0

def load_sample():
    window = []
    all_read = True
    window_fnames = []

    for frame_id in range(start_id, start_id + syncnet_T):
        frame = join(vid_dir, '{}.jpg'.format(frame_id))
        window_fnames.append(frame)

    for fname in window_fnames:
        img = cv2.imread(fname)
        if img is None:
            all_read = False
            break
        try:
            img = cv2.resize(img, (hparams.img_size, hparams.img_size))
        except Exception as e:
            all_read = False
            break

        window.append(img)

    def crop_audio_window(spec, start_frame_num):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx: end_idx, :]

    audio_path = f'../datasets-local/audios-flac/{vid_name}.flac'
    # wavpath = join(vidname, "audio.wav")
    wav = audio.load_wav(audio_path, hparams.sample_rate)
    orig_mel = audio.melspectrogram(wav).T

    mel = crop_audio_window(orig_mel.copy(), 0)
    assert mel.shape[0] == syncnet_mel_step_size

    # H x W x 3 * T
    x = np.concatenate(window, axis=2) / 255.
    x = x.transpose(2, 0, 1)
    x = x[:, x.shape[1]//2:]

    x = torch.FloatTensor(x)
    mel = torch.FloatTensor(mel.T).unsqueeze(0)
    return x, mel


x_batch = []
mel_batch = []
for k in tqdm(range(4)):
    x, mel = load_sample()
    x_batch.append(x.unsqueeze(0))
    mel_batch.append(mel.unsqueeze(0))

x_batch = torch.cat(x_batch)
mel_batch = torch.cat(mel_batch)

print('IMG', x_batch.shape)
print('MEL', mel_batch.shape)

audio_embed = model.audio_encoder(mel_batch)
face_embed = model.face_encoder(x_batch)

print('A-EMBED', audio_embed.shape)
print('F-EMBED', face_embed.shape)