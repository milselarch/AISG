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

fake_hist = []
real_hist = []
scores = []

fakes = df[df['label'] == 1]
reals = df[df['label'] == 0]
fake_filenames = fakes['filename'].to_numpy().tolist()
real_filenames = reals['filename'].to_numpy().tolist()
fake_filenames = sorted(fake_filenames)
real_filenames = sorted(real_filenames)

print(fake_filenames)
print(real_filenames)

print(len(df))
filenames = df['filename'].to_numpy().tolist()
filenames = filenames[:10]
print(len(df))

cache = {}


def resolve(filename, cache_mel=True):
    try:
        name = filename[:filename.index('.')]
    except ValueError as e:
        print('bad filename', filename)
        raise e

    file_path = f'../datasets-local/audios-flac/{name}.flac'

    if not os.path.exists(file_path):
        invalid.append(file_path)
        raise FileNotFoundError

    if name in cache:
        return cache[name]

    mel, _, duration = utils.process(file_path, '', 0)
    mel = mel[:clip]

    if cache_mel:
        cache[name] = mel

    return mel


real_filenames = real_filenames[:]
fake_filenames = fake_filenames[:]

for real_filename in tqdm(real_filenames):
    try:
        real_mel = resolve(real_filename)
    except FileNotFoundError:
        continue

pbar = tqdm(range(len(fake_filenames)))

for k in pbar:
    fake_filename = fake_filenames[k]
    pbar.set_description(f'loading {k} {fake_filename}')

    try:
        fake_mel = resolve(fake_filename, cache_mel=False)
    except FileNotFoundError:
        continue

    real_iterations = 0

    for i in range(len(real_filenames)):
        real_filename = real_filenames[i]

        try:
            real_mel = resolve(real_filename)
        except FileNotFoundError:
            continue

        real_iterations += 1
        fake_hist.append(k)
        real_hist.append(i)
        distance = np.linalg.norm(real_mel - fake_mel)
        scores.append(distance)

    # print(real_iterations)

stats_df = pd.DataFrame(data={
    'fakes': fake_hist, 'reals': real_hist,
    'scores': scores
})

stats_df.to_csv(f'csvs/aisg-stats-{stamp}.csv', index=False)
print('INVALID PATHS', invalid)