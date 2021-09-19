from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip

import random
import platform

from pdb import set_trace as bp

mel_step_size = 16
device = 'cuda'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

class obs(object):
    def __init__(self):
        self.checkpoint_path = 'pretrained/wav2lip.pth'
        self.outfile = 'results/result_voice.mp4'
        self.static = False

        self.fps = 25
        self.pads = [0, 10, 0, 0]
        self.face_det_batch_size = 16
        self.wav2lip_batch_size = 128
        self.resize_factor = 1

        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.rotate = False

        self.nosmooth = False
        self.face = '../datasets/train/videos/fd1404019e28214f.mp4'
        self.audio = '../datasets/train/videos/0b37e062f7751dac.mp4'
        self.img_size = 96


args = obs()

video_file = '../datasets/train/videos/fd1404019e28214f.mp4'
audio_file = '../datasets/train/videos/0b37e062f7751dac.mp4'
model_file = 'pretrained/wav2lip.pth'


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage
        )

    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}

    for k, v in s.items():
        new_s[k.replace('module.', '')] = v

    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()


# args = parser.parse_args()
# args.img_size = 96

if not audio_file.endswith('.wav'):
    print('Extracting raw audio...')
    command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_file, 'temp/temp.wav')

    subprocess.call(command, shell=True)
    audio_file = 'temp/temp.wav'

wav = audio.load_wav(audio_file, 16000)
mel = audio.melspectrogram(wav)
print(mel.shape)

video_stream = cv2.VideoCapture(video_file)
fps = video_stream.get(cv2.CAP_PROP_FPS)
frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
print('FRAMES IS', frames)
print('FPS IS', fps)
assert fps != 0


if np.isnan(mel.reshape(-1)).sum() > 0:
    raise ValueError('Mel contains nan! Using a TTS voice?')

mel_chunks = []
mel_idx_multiplier = 80. / fps
i = 0

while 1:
    start_idx = int(i * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
        mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
        break

    mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
    i += 1

model = load_model(model_file)
audio_pred = None

def random_sample_mel_chunks(
    mel_chunks, batch_size=None, transpose=True
):
    if batch_size is None:
        batch_size = args.wav2lip_batch_size

    index_range = range(0, len(mdels) - batch_size)
    index = random.choice(index_range)

    sub_mel_batch = mel_chunks[index: index + batch_size]
    sub_mel_batch = np.asarray(sub_mel_batch)
    sub_mel_batch = np.reshape(sub_mel_batch, [
        len(sub_mel_batch), sub_mel_batch.shape[1],
        sub_mel_batch.shape[2], 1
    ])

    if transpose:
        sub_mel_batch = np.transpose(sub_mel_batch, (0, 3, 1, 2))

    return sub_mel_batch


def datagen(mel_chunks):
    sub_mel_batch = []

    for i, m in enumerate(mel_chunks):
        # print(f'ENUM {i, m}')
        sub_mel_batch.append(m)

        if len(sub_mel_batch) >= args.wav2lip_batch_size:
            sub_mel_batch = np.asarray(sub_mel_batch)
            sub_mel_batch = np.reshape(sub_mel_batch, [
                len(sub_mel_batch), sub_mel_batch.shape[1],
                sub_mel_batch.shape[2], 1
            ])

            yield sub_mel_batch
            sub_mel_batch = []

    if len(sub_mel_batch) > 0:
        sub_mel_batch = np.asarray(sub_mel_batch)
        sub_mel_batch = np.reshape(sub_mel_batch, [
            len(sub_mel_batch), sub_mel_batch.shape[1],
            sub_mel_batch.shape[2], 1
        ])

        yield sub_mel_batch


batch_size = args.wav2lip_batch_size
total = int(np.ceil(float(len(mel_chunks)) / batch_size))
gen = datagen(mel_chunks)
pbar = tqdm(gen, total=total)

for i, mel_batch in enumerate(pbar):
    print(mel_batch.shape)
    np_mel_batch = np.transpose(mel_batch, (0, 3, 1, 2))
    mel_batch = torch.FloatTensor(np_mel_batch).to(device)

    with torch.no_grad():
        audio_pred = model.encode_audio(mel_batch)
        print('AUDIO PRED', audio_pred.shape)
        print(audio_pred)

    continue
