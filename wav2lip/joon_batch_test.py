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

batch_path = 'temp/test-batch.npy'
# preload_path = 'temp/test-model.pt'
preload_path = 'saves/checkpoints/211125-0108/E1261472_T0.6_V0.54.pt'

extractor = NeuralFaceExtract()
trainer = SyncnetTrainer(
    preload_path=preload_path,
    use_cuda=True, load_dataset=False, use_joon=True,
    old_joon=False, pred_ratio=1.0, is_checkpoint=False
)

device = torch.device("cuda")
n_batch = np.load(
    open(batch_path, 'rb'), allow_pickle=True
)[()]

audio_batch = n_batch['mels']
image_batch = n_batch['images']
torch_image_batch = FloatTensor(image_batch).to(device)
torch_audio_batch = FloatTensor(audio_batch).to(device)

trainer.model.eval()
with torch.no_grad():
    torch_preds = trainer.model.pred_fcc(
        torch_audio_batch, torch_image_batch, fcc_ratio=1.0
    )

predictions = torch_preds.detach().cpu().numpy().flatten()

mean = np.mean(predictions)
median = np.median(predictions)

print(f'PRED LIST', predictions.tolist())
print(f'PREDICTIONS', predictions)
print(f'MEAN = {mean}')
print(f'MEDIAN = {median}')

print('DONE')