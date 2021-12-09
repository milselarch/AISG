import random

import ParentImport

import os
import audio
import pandas as pd
import numpy as np
import cProfile
import random
import gc

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from hparams import hparams
from datetime import datetime
from NeuralFaceExtract import NeuralFaceExtract
from SyncnetTrainer import SyncnetTrainer
from BaseDataset import BaseDataset
from BaseDataset import MelCache

base_dir = 'saves/checkpoints'
preload_path = f'{base_dir}/211125-1900/E6143040_T0.77_V0.66.pt'

trainer = SyncnetTrainer(
    use_cuda=True, load_dataset=False, use_joon=True,
    # preload_path=preload_path, is_checkpoint=False,
    preload_path=preload_path, old_joon=False, pred_ratio=1.0,
    is_checkpoint=False,

    fcc_list=(512, 128, 32), dropout_p=0.5,
    transform_image=True
)

df = pd.read_csv('../stats/all-labels.csv')
all_filenames = df['filename'].to_numpy()
is_swap_fake = df['swap_fake'] == 1
is_real = df['label'] == 0

real_files = df[is_real]['filename']
swap_fakes = df[is_swap_fake]['filename']
real_files = real_files.to_numpy().tolist()
swap_fakes = swap_fakes.to_numpy().tolist()

filenames = real_files + swap_fakes
filename = filenames[0]
print(f'target filename {filename}')

mel_cache = MelCache()
mel_cache_path = 'saves/preprocessed/mel_cache_all.npy'
mel_cache.preload(mel_cache_path)

dataset, dataloader = trainer.make_data_loader(
    file_map=filenames, mel_cache=mel_cache
)

random.seed(42)
load_result = dataset.load_face_image_map(filename)
face_image_map, num_faces = load_result
face_no = face_image_map.face_nos[0]

face_samples = face_image_map.sample_face_frames(
    face_no, consecutive_frames=5, extract=False,
    max_samples=1
)

cct = mel_cache[filename]
face_sub_samples = face_samples[:1]
sample_img_batch, sample_mel_batch = trainer.to_torch_batch(
    face_sub_samples, cct, fps=face_image_map.fps,
    is_raw_audio=False, to_device=False,
    auto_double=False
)

sample_imgs = sample_img_batch
sample_mels = sample_mel_batch

torch_imgs, torch_mels = dataset.choose_sample(
    filename=filename, face_no=face_no,
    frame_no=face_sub_samples[0][0].frame_no,
    mirror_prob=0
)
print(f'face samples {face_samples}')
print('DONE')