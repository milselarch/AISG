import numpy as np
import pandas as pd

from tqdm.auto import tqdm

df = pd.read_csv('stats/face-predictions-211215-2244.csv')
filenames = df['filename'].to_numpy()
face_log = df['face'].to_numpy()
pred_log = df['pred'].to_numpy()

agg = {}

for k in tqdm(range(len(filenames))):
    filename = filenames[k]
    face = face_log[k]
    pred = pred_log[k]

    if filename not in agg:
        agg[filename] = {face: []}

    file_agg = agg[filename]
    if face not in file_agg:
        file_agg[face] = []

    file_agg[face].append(pred)

collate_preds = {}

for filename in agg:
    file_agg = agg[filename]
    face_preds = []

    for face in file_agg:
        face_preds = file_agg[face]
        pred = np.median(face_preds)
        face_preds.append(pred)

    file_pred = max(face_preds)
    collate_preds[filename] = file_pred

out_df = pd.DataFrame(data={
    'filename': list(collate_preds.keys()),
    'preds': list(collate_preds.values())
})

out_path = 'stats/effnet_preds.csv'
out_df.to_csv(out_path, index=False)
print(f'exported to {out_path}')