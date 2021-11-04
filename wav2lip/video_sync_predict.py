import ParentImport

import os
import audio
import pandas as pd
import numpy as np

from hparams import hparams
from datetime import datetime
from NeuralFaceExtract import NeuralFaceExtract
from SyncnetTrainer import SyncnetTrainer

class VideoSyncPredictor(object):
    def __init__(self):
        self.extractor = NeuralFaceExtract()
        self.trainer = SyncnetTrainer(
            use_cuda=False, load_dataset=False,
            preload_path='pretrained/lipsync_expert.pth',
            is_checkpoint=True, strict=False
        )

        self.audio_base_dir = '../datasets-local/audios-flac'

        df = pd.read_csv('../stats/all-labels.csv')
        all_filenames = df['filename'].to_numpy()
        is_swap_fake = df['swap_fake'] == 1
        is_real = df['label'] == 0

        self.real_files = df[is_real]['filename'].to_numpy()
        self.swap_fakes = df[is_swap_fake]['filename'].to_numpy()
        print('SWAPS', self.swap_fakes[:5], self.swap_fakes.shape)
        print('REALS', self.real_files[:5], self.real_files.shape)

        self.filenames = all_filenames
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
        wav = audio.load_wav(audio_path, hparams.sample_rate)
        orig_mel = audio.melspectrogram(wav).T
        num_faces = len(face_image_map)

        for face_no in face_image_map:
            face_samples = face_image_map.sample_face_frames(
                face_no, consecutive_frames=5, extract=False
            )
            predictions = self.trainer.face_predict(
                face_samples, orig_mel, fps=face_image_map.fps,
                to_numpy=True
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