import pandas as pd
import numpy as np
import os

from utils import hparams
from trainer import Trainer
from tqdm.auto import tqdm

df = pd.read_csv('../datasets/extra-labels.csv')
filenames = df['filename'].to_numpy().tolist()
for filename in filenames:
    assert filename.endswith('.mp4')

invalid = []
roll = 20

print(df)
# print('FILENAMES', filenames)
# input('>>> ')

trainer = Trainer(cache_threshold=20, load_dataset=False)
path = './saves/checkpoints/210928-1400/E35392_T0.86_V0.68.pt'
trainer.load_model(path)

real_test_names = open('aisg-test.txt').read().split('\n')
df['trained_upon'] = 0
df['median_pred'] = 0.5
df['mean_pred'] = 0.5
df['3rd_quartile_pred'] = 0.5
df['1st_quartile_pred'] = 0.5
df['group_pred'] = 0.5

test_files = [f'{filename}.mp4' for filename in real_test_names]
# eval_files = filenames[10:20] + test_files[0:10]
eval_files = filenames

# print('EVAL', eval_files)
pbar = tqdm(eval_files)
# pbar = tqdm(filenames[-10:])

for filename in pbar:
    name = filename[:filename.index('.')]
    file_path = f'../datasets-local/audios-flac/{name}.flac'
    pbar.set_description(name)

    if not os.path.exists(file_path):
        invalid.append(file_path)
        continue

    print(filename)
    cond = df['filename'] == filename
    row = df[cond]
    label = row['label'].to_numpy()[0]

    if (label == 0) and (name not in real_test_names):
        df.loc[cond, 'trained_upon'] = 1

    preds = trainer.batch_predict(file_path)
    preds = preds.flatten()
    # print('PREDS', preds, len(preds))

    mean_pred = np.mean(preds)
    median_pred = np.median(preds)
    roll_pred = pd.Series(preds).rolling(roll).median()
    roll_pred = roll_pred.to_numpy()
    roll_pred = roll_pred[~np.isnan(roll_pred)]

    group_pred = np.percentile(sorted(roll_pred), 75)
    quartile_pred_3 = np.percentile(sorted(preds), 75)
    quartile_pred_1 = np.percentile(sorted(preds), 25)

    df.loc[cond, 'mean_pred'] = median_pred
    df.loc[cond, 'median_pred'] = median_pred
    df.loc[cond, '3rd_quartile_pred'] = quartile_pred_3
    df.loc[cond, '1st_quartile_pred'] = quartile_pred_1
    df.loc[cond, 'group_pred'] = group_pred
    print(df.loc[cond])

df.to_csv('csvs/aisg-preds.csv', index=False)
print('INVALID PATHS', invalid)