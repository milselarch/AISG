import random

import ParentImport

import os
import sys
import audio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cProfile
import torch
import gc

from hparams import hparams
from tqdm.auto import tqdm
from datetime import datetime

from FaceSamplesHolder import FaceSamplesHolder
from NeuralFaceExtract import NeuralFaceExtract
from SyncnetTrainer import SyncnetTrainer
from BaseDataset import BaseDataset
from BaseDataset import MelCache

seed = 42
face_batch_size = 32
use_cuda = True
checkpoint_dir = '../saves/checkpoints'
preload_path = f'{checkpoint_dir}/211202-0328/E8554752_T0.78_V0.68.pt'

extractor = NeuralFaceExtract()

mel_cache = MelCache()
mel_cache_path = '../saves/preprocessed/mel_cache_all.npy'
mel_cache.preload(mel_cache_path)

video_base_dir = '../../datasets/train/videos'
face_base_dir = '../../datasets/extract/mtcnn-lip'
df = pd.read_csv('../../stats/all-labels.csv')
all_filenames = df['filename'].to_numpy()

is_real = df['label'] == 0
is_swap_fake = df['swap_fake'] == 1
real_files = df[is_real]['filename']
swap_fakes = df[is_swap_fake]['filename']
real_files = real_files.to_numpy().tolist()
swap_fakes = swap_fakes.to_numpy().tolist()
filenames = real_files + swap_fakes

random.seed(seed)
random.shuffle(filenames)
filename = filenames[0]

trainer = SyncnetTrainer(
    face_base_dir=face_base_dir,
    use_cuda=use_cuda, load_dataset=False,
    use_joon=True, old_joon=False,
    # preload_path=preload_path, is_checkpoint=False,
    preload_path=preload_path,

    fcc_list=(512, 128, 32),
    pred_ratio=1.0, dropout_p=0.5,
    is_checkpoint=False, predict_confidence=True,
    transform_image=True, eval_mode=False
)

dataset, dataloader = trainer.make_data_loader(
    file_map=[filename], mel_cache=mel_cache,
    face_path='../../stats/all-labels.csv',
    labels_path='../../datasets/train.csv',
    detect_path='../../stats/mtcnn/labelled-mtcnn.csv',
    filter_talker=False
)

"""
vid_image_map = self.extractor.process_filepath(
    filename, batch_size=face_batch_size, every_n_frames=1,
    skip_detect=10, ignore_detect=5,
    export_size=256, base_dir=video_base_dir
)
"""

vid_image_map = extractor.process_filepath(
    filename, base_dir=video_base_dir,
    batch_size=face_batch_size, every_n_frames=1,
    skip_detect=5, ignore_detect=20,
    export_size=224, displace_next=2
)

ds_extraction = dataset.load_face_image_map(filename)
ds_image_map, num_faces = ds_extraction
print('IMAGE MAPS EXTRACTED')