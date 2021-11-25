import ParentImport

import os
import audio
import torch
import random
import pandas as pd
import numpy as np

from hparams import hparams
from torch import FloatTensor
from datetime import datetime
from NeuralFaceExtract import NeuralFaceExtract
from SyncnetTrainer import SyncnetTrainer

# preload_path = 'saves/checkpoints/211120-1303/E2558336_T0.73_V0.65.pt'
preload_path = 'saves/checkpoints/211125-0108/E1261472_T0.6_V0.54.pt'
# preload_path = 'saves/checkpoints/211120-1303/E2558336_T0.73_V0.65.pt'

extractor = NeuralFaceExtract()
trainer = SyncnetTrainer(
    preload_path=preload_path,
    use_cuda=True, load_dataset=False, use_joon=True,
    old_joon=False, pred_ratio=1.0, is_checkpoint=False
)

trainer.load_parameters(preload_path)

trainer.dataset.load_datasets()
file_map = trainer.dataset.train_face_files
# file_map = ['c59d2549456ad02a.mp4']
dataset, dataloader = trainer.make_data_loader(file_map=file_map)
file_map = dataset.file_map
filepaths = list(file_map.keys())

# c59d2549456ad02a
random.seed(20)

name = filepaths[0]
# filename = f'{name}.mp4'
# filename = '0ae8d15530450892.mp4'
filename = '0c54cf96a5f72f11.mp4'
print(f'predicting {filename}')

image_paths = file_map[name]
image_batch, audio_batch = [], []
batch_image_paths = []

for k in range(16):
    image_path = random.choice(image_paths)
    batch_image_paths.append(image_path)

    frame_no = dataset.get_frame_no(image_path)
    torch_imgs, torch_mels = dataset.choose_sample(
        filename=filename, frame_no=frame_no
    )

    image_batch.append(torch_imgs)
    audio_batch.append(torch_mels)

device = torch.device("cuda")
torch_image_batch = torch.cat(image_batch, dim=0).to(device)
torch_audio_batch = torch.cat(audio_batch, dim=0).to(device)
torch_preds = trainer.model.pred_fcc(
    torch_audio_batch, torch_image_batch, fcc_ratio=1.0
)

predictions = torch_preds.detach().cpu().numpy().flatten()

mean = np.mean(predictions)
median = np.median(predictions)
quartile_pred_1 = np.percentile(sorted(predictions), 25)

print(f'PREDICTIONS', predictions)
print(f'MEAN = {mean}')
print(f'MEDIAN = {median}')
print(f'Q1 = {quartile_pred_1}')

print('DONE')