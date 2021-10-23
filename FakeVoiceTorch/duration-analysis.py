import utils
import pandas as pd
import numpy as np
import os

from utils import hparams
from trainer import Trainer
from tqdm.auto import tqdm

invalid = []
clip = 80

stamp = Trainer.make_date_stamp()
df = pd.read_csv('../datasets/extra-labels.csv')
filenames = df['filename'].to_numpy().tolist()
df['duration'] = 0
duration_map = {}

print(df)
pbar = tqdm(filenames)

for filename in pbar:
    name = filename[:filename.index('.')]
    file_path = f'../datasets-local/audios-flac/{name}.flac'

    if not os.path.exists(file_path):
        continue

    duration = utils.get_duration(file_path, '')
    cond = df['filename'] == filename
    label = df[cond]['label'].to_numpy()[0]

    # print('DURATION', duration)
    df.loc[cond, 'duration'] = duration
    desc = f'{filename} - {duration}'
    pbar.set_description(desc)

    if duration not in duration_map:
        duration_map[duration] = {'real': [], 'fake': []}

    str_label = 'fake' if label == 1 else 'real'
    duration_map[duration][str_label].append(filename)

# df.to_csv(f'csvs/aisg-durations-{stamp}.csv', index=False)
print('INVALID PATHS', invalid)
print('END')