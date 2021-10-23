import utils
import pandas as pd
import numpy as np
import os

from utils import hparams
from trainer import Trainer
from tqdm.auto import tqdm

invalid = []

stamp = Trainer.make_date_stamp()
df = pd.read_csv('csvs/aisg-durations-210929-0931.csv')
filenames = df['filename'].to_numpy().tolist()
labels = df['label'].to_numpy()
durations = df['duration'].to_numpy()

duration_map = {}

print(df)
pbar = tqdm(range(len(filenames)))

for k in pbar:
    filename = filenames[k]
    duration = durations[k]
    label = labels[k]

    name = filename[:filename.index('.')]
    file_path = f'../datasets-local/audios-flac/{name}.flac'

    if not os.path.exists(file_path):
        continue

    # print('DURATION', duration)
    # df.loc[cond, 'duration'] = duration
    desc = f'{filename} - {duration}'
    pbar.set_description(desc)

    if duration not in duration_map:
        duration_map[duration] = {'real': [], 'fake': []}

    str_label = 'fake' if label == 1 else 'real'
    duration_map[duration][str_label].append(filename)


reals_only = {}
fakes_only = {}
both_real_fake = {}

for duration in duration_map:
    files = duration_map[duration]

    if len(files['real']) == 0:
        fakes_only[duration] = files
        continue

    if len(files['fake']) == 0:
        reals_only[duration] = files
        continue

    both_real_fake[duration] = files

# sum([len(both_real_fake[dur]['fake']) for dur in both_real_fake])
print('REALS ONLY', len(reals_only))
print('FAKES ONLY', len(fakes_only))
print('BOTH ONLY', len(both_real_fake))
print('INVALID PATHS', invalid)
print('END')