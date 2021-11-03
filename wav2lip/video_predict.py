import ParentImport

import audio
import pandas as pd

from hparams import hparams
from NeuralFaceExtract import NeuralFaceExtract
from SyncnetTrainer import SyncnetTrainer

trainer = SyncnetTrainer(
    use_cuda=False, load_dataset=False
)

df = pd.read_csv('../stats/all-labels.csv')
swap_fakes = df[df['swap_fake'] == 1]['filename'].to_numpy()
real_files = df[df['label'] == 0]['filename'].to_numpy()

print('SWAPS', swap_fakes[:5], swap_fakes.shape)
print('REALS', real_files[:5], real_files.shape)

filenames = swap_fakes[:1]
assert len(filenames) == 1

def on_faces_loaded(name, face_image_map, pbar):
    if '.' in name:
        name = name[:name.index('.')]

    audio_base_dir = '../datasets-local/audios-flac'
    audio_path = f'{audio_base_dir}/{name}.flac'
    wav = audio.load_wav(audio_path, hparams.sample_rate)
    orig_mel = audio.melspectrogram(wav).T

    for face_no in face_image_map:
        face_samples = face_image_map.sample_face_frames(
            face_no, consecutive_frames=5
        )
        predictions = trainer.face_predict(
            face_samples, orig_mel, fps=face_image_map.fps
        )


extractor = NeuralFaceExtract()
extractor.process_filepaths(
    filenames, every_n_frames=1, skip_detect=10,
    export_size=96, callback=on_faces_loaded
)


