import random

import ParentImport

import os
import sys
import audio
import pandas as pd
import numpy as np
import cProfile
import torch
import copy
import gc

from tqdm.auto import tqdm

from hparams import hparams
from datetime import datetime
from FaceSamplesHolder import FaceSamplesHolder
from NeuralFaceExtract import NeuralFaceExtract
from SyncnetTrainer import SyncnetTrainer
from BaseDataset import BaseDataset
from BaseDataset import MelCache

# preload_path = 'saves/checkpoints/211110-0127/E930304_T0.9_V0.91.pt'
# preload_path = 'saves/checkpoints/211112-1504/E1124960_T0.64_V0.13.pt'
# preload_path = 'pretrained/syncnet_joon.model'
# preload_path = 'saves/checkpoints/211120-1303/E2558336_T0.73_V0.65.pt'
# preload_path = 'saves/checkpoints/211125-0108/E1261472_T0.6_V0.54.pt'
# preload_path = 'saves/checkpoints/211125-0108/E1178048_T0.6_V0.52.pt'
# preload_path = 'saves/checkpoints/211125-1900/E6143040_T0.77_V0.66.pt'
# preload_path = 'saves/checkpoints/211125-1900/E2674624_T0.68_V0.56.pt'
# preload_path = 'saves/checkpoints/211202-0328/E8554752_T0.78_V0.68.pt'
# preload_path = 'saves/checkpoints/211207-0123/E6668864_T0.9_V0.83.pt'
preload_path = 'saves/checkpoints/211202-0328/E10695968_T0.84_V0.69.pt'
# preload_path = 'saves/checkpoints/211125-1900/E6143040_T0.77_V0.66.pt'

class VideoSyncPredictor(object):
    def __init__(
        self, seed=42, use_cuda=True, mtcnn_cuda=True,
        face_base_dir='../datasets/extract/mtcnn-sync',
        use_mouth_image=True
    ):
        self.extractor = NeuralFaceExtract(
            use_cuda=mtcnn_cuda
        )
        self.trainer = SyncnetTrainer(
            face_base_dir=face_base_dir,
            use_cuda=use_cuda, load_dataset=False,
            use_joon=True, old_joon=False,
            # preload_path=preload_path, is_checkpoint=False,
            preload_path=preload_path,

            fcc_list=(512, 128, 32),
            pred_ratio=1.0, dropout_p=0.5,
            is_checkpoint=False, predict_confidence=True,
            transform_image=True, eval_mode=True
        )

        self.trainer.model.disable_norm_toggle()

        self.mtcnn_cuda = mtcnn_cuda
        self.use_mouth_image = use_mouth_image
        # self.trainer.model.disable_norm_toggle()
        self.face_base_dir = face_base_dir
        self.audio_base_dir = '../datasets/extract/audios-flac'
        self.video_base_dir = '../datasets/train/videos'

        df = pd.read_csv('../stats/all-labels.csv')
        all_filenames = df['filename'].to_numpy()
        is_swap_fake = df['swap_fake'] == 1
        is_real = df['label'] == 0

        real_files = df[is_real]['filename']
        swap_fakes = df[is_swap_fake]['filename']
        self.real_files = real_files.to_numpy().tolist()
        self.swap_fakes = swap_fakes.to_numpy().tolist()
        self.all_filenames = all_filenames
        self.use_cuda = use_cuda
        self.seed = seed

        print(f'num files loaded {all_filenames}')
        print('SWAPS', self.swap_fakes[:5], len(self.swap_fakes))
        print('REALS', self.real_files[:5], len(self.real_files))

        # self.filenames = self.swap_fakes
        self.train_files = open('stats/train.txt').read().split('\n')
        self.test_files = open('stats/test.txt').read().split('\n')

        self.filenames = self.real_files + self.swap_fakes
        # self.filenames = all_filenames

        random.seed(seed)
        random.shuffle(self.filenames)

        # self.filenames = ['c59d2549456ad02a.mp4']  # all_filenames
        # random.shuffle(self.filenames)
        # self.filenames = ['d27c1c217aae3e70.mp4']
        # self.filenames = self.swap_fakes[:3]

        self.face_log = []
        self.mean_preds = []
        self.filename_log = []

        self.std_preds = []
        self.median_preds = []
        self.quartile_preds_3 = []
        self.quartile_preds_1 = []
        self.max_preds = []
        self.min_preds = []

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

        print(f'LOADED {name}')
        tag = self.get_tag(filename)
        audio_path = f'{self.audio_base_dir}/{name}.flac'
        raw_audio = audio.load_wav(audio_path, hparams.sample_rate)
        num_faces = len(face_image_map)

        for face_no in face_image_map:
            face_samples = face_image_map.sample_face_frames(
                face_no, consecutive_frames=5, extract=False
            )
            predictions = self.trainer.face_predict_joon(
                face_samples, raw_audio, fps=face_image_map.fps,
                to_numpy=True, is_raw_audio=True,
                use_mouth_image=self.use_mouth_image
            )
            self.record_preds(
                predictions, face_no, num_faces,
                filename=filename, tag=tag
            )

    def record_preds(
        self, predictions, face_no, num_faces, filename, tag=''
    ):
        mean_pred = np.mean(predictions)
        median_pred = np.median(predictions)
        quartile_pred_3 = np.percentile(sorted(predictions), 75)
        quartile_pred_1 = np.percentile(sorted(predictions), 25)

        std_pred = np.std(predictions)
        max_pred = np.max(predictions)
        min_pred = np.min(predictions)

        self.face_log.append(face_no)
        self.quartile_preds_3.append(quartile_pred_3)
        self.quartile_preds_1.append(quartile_pred_1)
        self.filename_log.append(filename)
        self.median_preds.append(median_pred)
        self.mean_preds.append(mean_pred)

        self.std_preds.append(std_pred)
        self.max_preds.append(max_pred)
        self.min_preds.append(min_pred)

        # print('PREDS', face_no, predictions)
        header = f'[{face_no}/{num_faces - 1}][{tag}]'

        print(f'NUM PREDICTIONS = {len(predictions)}')
        print(f'predicting {filename} {header}')
        print(f'3rd quartile pred: {quartile_pred_3}')
        print(f'1st quartile pred: {quartile_pred_1}')
        print(f'median pred: {median_pred}')
        print(f'mean pred: {mean_pred}')

        print(f'std pred: {std_pred}')
        print(f'min pred: {min_pred}')
        print(f'max pred: {max_pred}')

    def profile_infer_videos(
        self, *args, profile_dir='saves/profiles', **kwargs
    ):
        stamp = self.make_date_stamp()
        assert os.path.isdir(profile_dir)
        profile_path = f'{profile_dir}/joon-pred-{stamp}.profile'
        profile = cProfile.Profile()
        profile.enable()

        try:
            self.infer_videos(*args, **kwargs)
        except Exception as e:
            print('TRAINING FAILED')
            raise e
        finally:
            profile.disable()
            profile.dump_stats(profile_path)
            print(f'profile saved to {profile_path}')

    def profile_infer(
        self, *args, profile_dir='saves/profiles', **kwargs
    ):
        stamp = self.make_date_stamp()
        assert os.path.isdir(profile_dir)
        profile_path = f'{profile_dir}/joon-pred-{stamp}.profile'
        profile = cProfile.Profile()
        profile.enable()

        try:
            self.quick_infer(*args, **kwargs)
        except Exception as e:
            print('TRAINING FAILED')
            raise e
        finally:
            profile.disable()
            profile.dump_stats(profile_path)
            print(f'profile saved to {profile_path}')

    def get_tag(self, filename):
        if filename in self.real_files:
            tag = 'R'
        elif filename in self.swap_fakes:
            tag = 'S'
        else:
            tag = 'F'

        return tag

    def quick_infer(
        self, filenames=None, clip=None, batch_size=16,
        max_samples=32
    ):
        if filenames is None:
            filenames = self.filenames
        if clip is not None:
            filenames = filenames[:clip]

        mel_cache = MelCache()
        mel_cache_path = 'saves/preprocessed/mel_cache_all.npy'
        mel_cache.preload(mel_cache_path)

        date_stamp = self.make_date_stamp()
        dataset, dataloader = self.trainer.make_data_loader(
            file_map=filenames, mel_cache=mel_cache
        )

        num_face_map = {}
        pbar = tqdm(filenames)
        samples_holder = FaceSamplesHolder(
            predictor=self.trainer, batch_size=batch_size
        )

        for filename in pbar:
            cct = mel_cache[filename]
            pbar.set_description(f'predicting {filename}')

            try:
                dataset.get_video_image_paths(filename)
            except KeyError as e:
                print(f'skipping {filename}')
                continue

            load_result = dataset.load_face_image_map(filename)
            face_image_map, num_faces = load_result
            num_face_map[filename] = num_faces

            for face_no in face_image_map:
                face_samples = face_image_map.sample_face_frames(
                    face_no, consecutive_frames=5, extract=False,
                    max_samples=max_samples
                )
                samples_holder.add_face_samples(
                    filename, face_samples=face_samples, mel=cct,
                    face_no=face_no, fps=face_image_map.fps
                )

            del face_image_map
            gc.collect()

        samples_holder.flush()
        return self._collate_samples(
            samples_holder, date_stamp=date_stamp,
            csv_tag='dataset'
        )

    def _collate_samples(
        self, samples_holder, date_stamp=None, csv_tag='vid-preds'
    ):
        if date_stamp is None:
            date_stamp = self.make_date_stamp()

        samples_holder.flush()
        preds_map = samples_holder.make_video_preds()
        video_preds_map, video_confs_map = preds_map
        all_preds, all_labels, all_confs = [], [], []
        video_preds, video_labels = [], []
        face_log, filename_log = [], []

        for filename in video_preds_map:
            current_video_preds = video_preds_map[filename]
            current_video_confs = video_confs_map[filename]
            print(f'\n video {filename}')
            face_preds = []

            for face_no in current_video_preds:
                predictions = current_video_preds[face_no]
                confidences = current_video_confs[face_no]
                num_faces = len(current_video_preds)

                np_preds = np.array(predictions)
                tag = self.get_tag(filename)

                print(f'[{face_no}] predictions: {np_preds}')

                self.record_preds(
                    predictions, face_no, num_faces,
                    filename=filename, tag=tag
                )

                label = tag != 'R'
                labels = [label] * len(predictions)

                all_labels.extend(labels)
                face_log.extend([face_no] * len(predictions))
                filename_log.extend([filename] * len(predictions))
                all_preds.extend(predictions)
                all_confs.extend(confidences)

                face_pred = np.median(predictions)
                print(f'[{face_no}] face pred: {face_pred}')
                face_preds.append(face_pred)

            video_pred = min(face_preds)
            tag = self.get_tag(filename)
            label = tag != 'R'

            video_preds.append(video_pred)
            video_labels.append(label)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        me, mse, acc = self.trainer.get_metrics(
            all_preds, all_labels
        )
        mean_pred = np.mean(all_preds)
        p_fake = sum(all_labels) / len(all_labels)

        print('')
        print('overall prediction stats')
        print(f'mean error: {me}')
        print(f'mean squared error: {mse}')
        print(f'mean pred: {mean_pred}')
        print(f'accuracy: {acc}')
        print(f'percent fake: {p_fake}')

        video_preds = np.array(video_preds)
        video_labels = np.array(video_labels)
        p_vid_fake = sum(video_labels) / len(video_labels)
        vid_me, vid_mse, vid_acc = self.trainer.get_metrics(
            video_preds, video_labels
        )

        print('')
        print(f'video mean error: {vid_me}')
        print(f'video mean squared error: {vid_mse}')
        print(f'video mean pred: {mean_pred}')
        print(f'video accuracy: {vid_acc}')
        print(f'percent vid fake: {p_vid_fake}')

        self.export_video_preds(date_stamp, tag=csv_tag)
        self.export_all_preds(
            all_preds, all_confs, face_log, filename_log,
            all_labels, date_stamp=date_stamp, tag=csv_tag
        )

        num_filenames = len(video_preds_map)
        predict_time_taken = samples_holder.timer.total
        predict_per_video = predict_time_taken / num_filenames
        print(f'predict time taken: {predict_time_taken}')
        print(f'predict time per video: {predict_per_video}')

    def export_all_preds(
        self, pred_log, conf_log, face_log, filename_log,
        label_log, date_stamp=None, tag='test'
    ):
        if date_stamp is None:
            date_stamp = self.make_date_stamp()

        df = pd.DataFrame(data={
            'filename': filename_log,
            'pred': pred_log, 'conf': conf_log,
            'face': face_log, 'label': label_log
        })

        export_path = f'stats/all-{tag}-{date_stamp}.csv'
        df.to_csv(export_path, index=False)
        print(f'all sync preds exported to {export_path}')

    def export_video_preds(self, date_stamp=None, tag='test'):
        if date_stamp is None:
            date_stamp = self.make_date_stamp()

        df = pd.DataFrame(data={
            'filename': self.filename_log,
            'mean_pred': self.mean_preds,
            'median_pred': self.median_preds,
            '1st_quartile_pred': self.quartile_preds_1,
            '3rd_quartile_pred': self.quartile_preds_3,

            'std_pred': self.std_preds,
            'max_pred': self.max_preds,
            'min_pred': self.min_preds,
            'face_no': self.face_log
        })

        export_path = f'stats/vid-{tag}-{date_stamp}.csv'
        df.to_csv(export_path, index=False)
        print(f'video sync preds exported to {export_path}')

    def start_mono(self, clip=None):
        date_stamp = self.make_date_stamp()
        filenames = copy.deepcopy(self.filenames)
        random.shuffle(filenames)

        if clip is not None:
            filenames = filenames[:clip]

        self.extractor.process_filepaths(
            filenames, every_n_frames=1,
            skip_detect=10, ignore_detect=5, export_size=224,
            callback=self.on_faces_loaded,
            base_dir='../datasets/train/videos'
        )

        self.export_video_preds(date_stamp)

    def infer_videos(
        self, filenames=None, clip=None, samples_batch_size=32,
        face_batch_size=32, use_mouth_image=None, rgb_to_bgr=False
    ):
        if use_mouth_image is None:
            use_mouth_image = self.use_mouth_image
        if filenames is None:
            filenames = self.filenames
        if clip is not None:
            filenames = filenames[:clip]

        date_stamp = self.make_date_stamp()
        samples_holder = FaceSamplesHolder(
            predictor=self.trainer, batch_size=samples_batch_size,
            use_mouth_image=use_mouth_image, rgb_to_bgr=rgb_to_bgr
        )

        mel_cache = MelCache()
        mel_cache_path = 'saves/preprocessed/mel_cache_all.npy'
        mel_cache.preload(mel_cache_path)
        pbar = tqdm(filenames)

        for filename in pbar:
            face_image_map = self.extractor.process_filepath(
                filename, batch_size=face_batch_size,
                every_n_frames=1, skip_detect=10, ignore_detect=5,
                export_size=256, base_dir='../datasets/train/videos'
            )

            cct = mel_cache[filename]

            for face_no in face_image_map:
                face_samples = face_image_map.sample_face_frames(
                    face_no, consecutive_frames=5, extract=False,
                    max_samples=32
                )
                samples_holder.add_face_samples(
                    filename, face_samples=face_samples, mel=cct,
                    face_no=face_no, fps=face_image_map.fps
                )

            del face_image_map
            gc.collect()

        samples_holder.flush()
        return self._collate_samples(
            samples_holder, date_stamp=date_stamp,
            csv_tag='orig'
        )


"""
[pre-scrambling]
211125-1900/E2674624_T0.68_V0.56.pt
---------------------------------
mean error: 0.6693221431075435
mean squared error: 0.7528739890437786
mean pred: 0.7487096786499023
accuracy: 0.31868131868131866
percent fake: 0.1978021978021978

211125-1900/E2674624_T0.68_V0.56.pt
---------------------------------
mean error: 0.18534135606346858
mean squared error: 5.503507536070516
mean pred: 0.3159324526786804
accuracy: 0.9282407407407407
percent fake: 0.24074074074074073

[mtcnn-sync on mtcnn-lip]
211125-1900/E2674624_T0.68_V0.56.pt
---------------------------------
mean error: 0.6087206145532912
mean squared error: 14.451363912977643
mean pred: 0.6550164818763733
accuracy: 0.375
percent fake: 0.24074074074074073

211202-0328/E8554752_T0.78_V0.68.pt [C32]
---------------------------------
video mean error: 0.39455374074168503
video mean squared error: 2.828592335567301
video mean pred: 0.5337652564048767
video accuracy: 0.625
percent vid fake: 0.1875
"""

if __name__ == '__main__':
    sync_predictor = VideoSyncPredictor(
        face_base_dir='../datasets/extract/mtcnn-lip',
        use_cuda=False, mtcnn_cuda=False,
        use_mouth_image=True
    )

    # sync_predictor.profile_infer(['07cc4dde853dfe59.mp4'])
    # sync_predictor.profile_infer(clip=32)
    # sync_predictor.profile_infer(batch_size=32, max_samples=None)
    # sync_predictor.profile_infer(clip=32, batch_size=32)
    # sync_predictor.profile_infer_videos(clip=32, rgb_to_bgr=True)
    sync_predictor.start_mono(clip=32)