import pandas as pd

from tqdm.auto import tqdm
from BaseDataset import BaseDataset
from SyncnetTrainer import SyncnetTrainer

trainer = SyncnetTrainer(
    use_cuda=True, load_dataset=True, use_joon=True,
    old_joon=False, pred_ratio=1.0, is_checkpoint=False
)

df = pd.read_csv('../datasets/train.csv')
all_filenames = df['filename'].to_numpy()

trainer.dataset.load_datasets()
# face_files = trainer.dataset.allowed_filenames
face_files = all_filenames

audio_base_dir = '../datasets/extract/audios-flac'
mel_cache = {}

for filename in tqdm(face_files):
    name = filename[:filename.index('.')]
    orig_mel = BaseDataset.load_audio_file(
        name, audio_base_dir, use_joon=True
    )

    mel_cache[name] = orig_mel

print('DONE')