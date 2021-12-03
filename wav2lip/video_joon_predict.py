import random

import ParentImport

import os
import sys
import audio
import pandas as pd
import numpy as np
import cProfile
import torch
import gc

from tqdm.auto import tqdm

from hparams import hparams
from datetime import datetime
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
preload_path = 'saves/checkpoints/211125-1900/E2674624_T0.68_V0.56.pt'

class FaceSamplesHolder(object):
    def __init__(self, predictor, batch_size=16):
        self.predictor = predictor
        self.batch_size = batch_size

        self.fps_cache = {}
        self.mel_cache = {}
        self.face_samples_cache = {}
        self.face_samples_map = {}
        self.face_preds_map = {}

    def add_face_sample(
        self, filename, face_samples, mel, face_no, fps
    ):
        key = (filename, face_no)
        self.mel_cache[filename] = mel
        self.face_samples_map[key] = face_samples
        self.fps_cache[filename] = fps
        has_predictions = True

        while has_predictions:
            has_predictions = self.auto_predict_samples(
                flush=False
            )

    def add_to_cache(self, key, face_sample):
        if len(self.face_samples_cache) < self.batch_size:
            if key not in self.face_samples_cache:
                self.face_samples_cache[key] = []

            cache_face_samples = self.face_samples_cache[key]
            cache_face_samples.append(face_sample)
            return True

        return False

    @staticmethod
    def warn(message):
        print(message, file=sys.stderr)

    def load_from_cache(self, num_samples, check_size=True):
        assert (
            num_samples <= len(self.face_samples_cache)
            or not check_size
        )

        img_batch, mel_batch = [], []
        init_cache_keys = list(self.face_samples_cache.keys())
        random.shuffle(init_cache_keys)
        cache_keys = init_cache_keys[::]

        if len(cache_keys) < num_samples:
            cache_size = len(cache_keys)
            self.warn(f'UNDERSIZED CACHE {cache_size}')

        while len(cache_keys) < num_samples:
            cache_keys += init_cache_keys

        for key in cache_keys[:num_samples]:
            filename, face_no = key
            face_samples = self.face_samples_cache[key]
            face_sample = random.choice(face_samples)

            cct = self.mel_cache[filename]
            fps = self.fps_cache[filename]
            t_img, t_mel = self.predictor.to_torch_batch(
                [face_sample], cct, fps=fps, auto_double=False
            )

            img_batch.append(t_img)
            mel_batch.append(t_mel)

        assert len(img_batch) == num_samples
        return img_batch, mel_batch

    def resolve_samples(self, check_size=True):
        length = len(self.face_samples_map)
        if check_size and (length < self.batch_size):
            return False

        img_batch, mel_batch, key_batch = [], [], []
        keys = list(self.face_samples_map.keys())

        for key in keys:
            filename, face_no = key
            face_samples = self.face_samples_map[key]
            face_sample = face_samples.pop()

            cct = self.mel_cache[filename]
            fps = self.fps_cache[filename]
            torch_data = self.predictor.to_torch_batch(
                [face_sample], cct, fps=fps, auto_double=False
            )

            if len(self.face_samples_map[key]) == 0:
                del self.face_samples_map[key]

            if torch_data is None:
                continue

            t_img, t_mel = torch_data
            self.add_to_cache(key, face_sample)
            if key not in self.face_samples_map:
                if key not in self.face_samples_cache:
                    del self.mel_cache[filename]

            key_batch.append(key)
            img_batch.append(t_img)
            mel_batch.append(t_mel)

        buffer_needed = self.batch_size - len(img_batch)
        cache_items = self.load_from_cache(
            buffer_needed, check_size=check_size
        )

        cache_img_batch, cache_mel_batch = cache_items
        img_batch.extend(cache_img_batch)
        mel_batch.extend(cache_mel_batch)

        t_img_batch = torch.cat(img_batch)
        t_mel_batch = torch.cat(mel_batch)
        return t_img_batch, t_mel_batch, key_batch

    def auto_predict_samples(self, flush=False):
        check_size = not flush
        torch_batch = self.resolve_samples(check_size)
        if torch_batch is False:
            return False

        t_img_batch, t_mel_batch, key_batch = torch_batch
        preds, confs = self.predictor.predict(t_mel_batch, t_img_batch)
        preds = preds.detach().cpu().numpy()

        for k, key in enumerate(key_batch):
            prediction = preds[k]
            if key not in self.face_preds_map:
                self.face_preds_map[key] = []

            face_preds = self.face_preds_map[key]
            face_preds.append(prediction)

        return True

    def flush(self):
        while len(self.face_samples_map) > 0:
            self.auto_predict_samples(flush=True)


class VideoSyncPredictor(object):
    def __init__(self, seed=42, use_cuda=True):
        self.extractor = NeuralFaceExtract()
        self.trainer = SyncnetTrainer(
            use_cuda=use_cuda, load_dataset=False, use_joon=True,
            # preload_path=preload_path, is_checkpoint=False,
            preload_path=preload_path, old_joon=False, pred_ratio=1.0,
            is_checkpoint=False,

            fcc_list=(512, 128, 32), dropout_p=0.5,
            transform_image=True
        )

        # self.trainer.model.disable_norm_toggle()
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
        self.train_files = open('train.txt').read().split('\n')
        self.test_files = open('test.txt').read().split('\n')

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
                to_numpy=True, is_raw_audio=True
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

    def profile_start(self, profile_dir='saves/profiles'):
        stamp = self.make_date_stamp()
        assert os.path.isdir(profile_dir)
        profile_path = f'{profile_dir}/joon-pred-{stamp}.profile'
        profile = cProfile.Profile()
        profile.enable()

        try:
            self.start()
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
        self, filenames=None, clip=None, batch_size=16
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
                    face_no, consecutive_frames=5, extract=False
                )
                samples_holder.add_face_sample(
                    filename, face_samples=face_samples, mel=cct,
                    face_no=face_no, fps=face_image_map.fps
                )

            del face_image_map
            gc.collect()

        samples_holder.flush()
        all_preds, all_labels = [], []
        face_preds_map = samples_holder.face_preds_map

        for key in face_preds_map:
            filename, face_no = key
            predictions = face_preds_map[key]
            np_preds = np.array(predictions)
            num_faces = num_face_map[filename]
            tag = self.get_tag(filename)
    
            print('')
            print(f'predictions: {np_preds}')

            self.record_preds(
                predictions, face_no, num_faces,
                filename=filename, tag=tag
            )

            labels = [tag != 'R'] * len(predictions)
            all_preds.extend(predictions)
            all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        me, mse, accuracy = self.trainer.get_metrics(
            all_preds, all_labels
        )
        mean_pred = np.mean(all_preds)
        p_fake = sum(all_labels) / len(all_labels)

        print('')
        print('overall prediction stats')
        print(f'mean error: {me}')
        print(f'mean squared error: {mse}')
        print(f'mean pred: {mean_pred}')
        print(f'accuracy: {accuracy}')
        print(f'percent fake: {p_fake}')
        self.store_preds(date_stamp)

    def store_preds(self, date_stamp=None):
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

        export_path = f'../stats/sync-vid-preds-{date_stamp}.csv'
        df.to_csv(export_path, index=False)
        print(f'sync preds exported to {export_path}')

    def start(self):
        date_stamp = self.make_date_stamp()

        self.extractor.process_filepaths(
            self.filenames, every_n_frames=1,
            skip_detect=10, ignore_detect=5, export_size=96,
            callback=self.on_faces_loaded,
            base_dir='../datasets/train/videos'
        )

        self.store_preds(date_stamp)


"""
mean error: 0.6693221431075435
mean squared error: 0.7528739890437786
mean pred: 0.7487096786499023
accuracy: 0.31868131868131866
percent fake: 0.1978021978021978
"""

if __name__ == '__main__':
    sync_predictor = VideoSyncPredictor(use_cuda=False)
    # sync_predictor.profile_infer(['07cc4dde853dfe59.mp4'])
    # sync_predictor.profile_infer(clip=32)
    sync_predictor.profile_infer(clip=16, batch_size=32)
    # sync_predictor.profile_start()