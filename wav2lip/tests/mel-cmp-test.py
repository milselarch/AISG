import ParentImport

import time
import numpy as np

from BaseDataset import MelCache
from itertools import product
from tqdm.auto import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

# np.linalg.norm
start = 100
clip_length = 100

mel_cache = MelCache()
mel_cache_path = '../saves/preprocessed/mel_cache_all.npy'
mel_cache.preload(mel_cache_path)

names = mel_cache.names
pbar = tqdm(product(names, names))
pbar.total = len(names) ** 2
start_time = time.perf_counter()

for name1, name2 in pbar:
    cct1, cct2 = mel_cache[name1], mel_cache[name2]
    cct_clip1 = cct1[:, :, :, start: start+clip_length]
    cct_clip2 = cct2[:, :, :, start: start+clip_length]

    closeness = cosine_similarity(cct_clip1, cct_clip2)

    # closeness = np.linalg.norm(cct_clip1 - cct_clip2)
    description = f'{name1} - {name2} closeness = {closeness}'
    pbar.set_description(description)

end_time = time.perf_counter()
duration = end_time - start_time
print(f'time taken: {duration}')

print(f'MEL CACHE: {mel_cache}')
print('DONE')