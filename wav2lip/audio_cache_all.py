import copy

from tqdm import tqdm
from SyncDataset import SyncDataset
from BaseDataset import BaseDataset

syncer = SyncDataset(load=False)
syncer.load_datasets()
base_dataset = BaseDataset(
    file_map=syncer.face_files,
    face_map=syncer.face_map
)

file_map = base_dataset.file_map

for filename in tqdm(syncer.train_face_files):
    name = filename[:filename.index('.mp4')]
    assert name in file_map
for filename in tqdm(syncer.test_face_files):
    name = filename[:filename.index('.mp4')]
    assert name in file_map

audio_cache = {}
for filename in tqdm(syncer.allowed_filenames):
    orig_mel = base_dataset.load_audio(filename)
    audio_cache[filename] = orig_mel

print('FILE MAP LENGTH', len(file_map))
print('FACE MAP LENGTH', len(syncer.face_map))

print('SYNC FACE FILES', len(syncer.face_files))
print('SYNC FACE MAP', len(syncer.face_files))

print('SYNC TRAIN FACE FILES', len(syncer.train_face_files))
print('SYNC TEST FACE FILES', len(syncer.test_face_files))
input('LOAD FINISHED >>> ')