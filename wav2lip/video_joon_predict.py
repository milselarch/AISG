import random

import ParentImport

import os
import audio
import pandas as pd
import numpy as np

from hparams import hparams
from datetime import datetime
from NeuralFaceExtract import NeuralFaceExtract
from SyncnetTrainer import SyncnetTrainer

# preload_path = 'saves/checkpoints/211110-0127/E930304_T0.9_V0.91.pt'
# preload_path = 'saves/checkpoints/211112-1504/E1124960_T0.64_V0.13.pt'

class VideoSyncPredictor(object):
    def __init__(self):
        self.extractor = NeuralFaceExtract()
        self.trainer = SyncnetTrainer(
            use_cuda=True, load_dataset=False, use_joon=True,
            # preload_path=preload_path, is_checkpoint=False,
            preload_path='pretrained/syncnet_joon.model',
            is_checkpoint=False
        )

        self.audio_base_dir = '../datasets/extract/audios-flac'

        df = pd.read_csv('../stats/all-labels.csv')
        all_filenames = df['filename'].to_numpy()
        is_swap_fake = df['swap_fake'] == 1
        is_real = df['label'] == 0

        real_files = df[is_real]['filename']
        swap_fakes = df[is_swap_fake]['filename']
        self.real_files = real_files.to_numpy().tolist()
        self.swap_fakes = swap_fakes.to_numpy().tolist()

        print('SWAPS', self.swap_fakes[:5], len(self.swap_fakes))
        print('REALS', self.real_files[:5], len(self.real_files))

        # self.filenames = self.swap_fakes
        self.train_files = open('train.txt').read().split('\n')
        self.test_files = open('test.txt').read().split('\n')

        # self.filenames = self.real_files + self.swap_fakes
        self.filenames = all_filenames
        random.shuffle(self.filenames)

        # self.filenames = ['c59d2549456ad02a.mp4']  # all_filenames
        # random.shuffle(self.filenames)
        # self.filenames = ['d27c1c217aae3e70.mp4']
        # self.filenames = self.swap_fakes[:3]

        self.face_log = []
        self.mean_preds = []
        self.filename_log = []
        self.median_preds = []
        self.quartile_preds_3 = []
        self.quartile_preds_1 = []

        # filenames = np.concatenate([swap_fakes[:1], real_files[:2]])
        # assert len(filenames) == 1

    @staticmethod
    def make_date_stamp():
        return datetime.now().strftime("%y%m%d-%H%M")

    def on_faces_loaded(
        self, filepath, face_image_map, pbar, img_filter=None
    ):
        name = os.path.basename(filepath)
        if '.' in name:
            name = name[:name.index('.')]

        filename = f'{name}.mp4'
        if face_image_map is None:
            print(f'BAD VIDEO {name}')
            return False

        if filename in self.real_files:
            tag = 'R'
        elif filename in self.swap_fakes:
            tag = 'S'
        else:
            tag = 'F'

        print(f'LOADED {name}')
        audio_path = f'{self.audio_base_dir}/{name}.flac'
        raw_audio = audio.load_wav(audio_path, hparams.sample_rate)
        num_faces = len(face_image_map)

        for face_no in face_image_map:
            face_samples = face_image_map.sample_face_frames(
                face_no, consecutive_frames=5, extract=False
            )
            predictions = self.trainer.face_predict_joon(
                face_samples, raw_audio, fps=face_image_map.fps,
                to_numpy=True, is_raw_audio=True
            )

            mean_pred = np.mean(predictions)
            median_pred = np.median(predictions)
            quartile_pred_3 = np.percentile(sorted(predictions), 75)
            quartile_pred_1 = np.percentile(sorted(predictions), 25)

            self.face_log.append(face_no)
            self.quartile_preds_3.append(quartile_pred_3)
            self.quartile_preds_1.append(quartile_pred_1)
            self.filename_log.append(filename)
            self.median_preds.append(median_pred)
            self.mean_preds.append(mean_pred)

            # print('PREDS', face_no, predictions)
            header = f'[{face_no}/{num_faces-1}][{tag}]'

            print(f'NUM PREDICTIONS = {len(predictions)}')
            print(f'predicting {filename} {header}')
            print(f'3rd quartile pred: {quartile_pred_3}')
            print(f'1st quartile pred: {quartile_pred_1}')
            print(f'median pred: {median_pred}')
            print(f'mean pred: {mean_pred}')

    def start(self):
        self.extractor.process_filepaths(
            self.filenames, every_n_frames=1,
            skip_detect=10, ignore_detect=5, export_size=96,
            callback=self.on_faces_loaded,
            base_dir='../datasets/train/videos'
        )

        df = pd.DataFrame(data={
            'filename': self.filename_log,
            'mean_pred': self.mean_preds,
            'median_pred': self.median_preds,
            '1st_quartile_pred': self.quartile_preds_1,
            '3rd_quartile_pred': self.quartile_preds_3,
            'face_no': self.face_log
        })

        date_stamp = self.make_date_stamp()
        export_path = f'../stats/sync-vid-preds-{date_stamp}.csv'
        df.to_csv(export_path, index=False)
        print(f'sync preds exported to {export_path}')


if __name__ == '__main__':
    sync_predictor = VideoSyncPredictor()
    sync_predictor.start()