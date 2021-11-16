from abc import ABC
from typing import Iterator

try:
    import ParentImport
    import audio

    from FaceAnalysis import FaceCluster
    from hparams import hparams
except ModuleNotFoundError:
    from . import ParentImport
    from . import audio

    from ..FaceAnalysis import FaceCluster
    from .hparams import hparams

import os
import sys
import random
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
import torch
import time
import cv2
import re

from tqdm.auto import tqdm
from torch.utils.data import IterableDataset
from datetime import datetime as Datetime


class MelCache(object):
    def __init__(self):
        self.cache = {}

    def __setitem__(self, filename, orig_mel):
        match = re.match('^[0-9a-z]{16}$', filename)
        assert match is not None
        assert type(orig_mel) is np.ndarray
        self.cache[filename] = orig_mel

    def __contains__(self, filename):
        name = filename
        if filename.endswith('.mp4'):
            name = name[:name.index('.')]

        return name in self.cache

    def __getitem__(self, filename):
        name = filename
        # print('GETITEM', name)

        if filename.endswith('.mp4'):
            name = name[:name.index('.')]

        return self.cache[name]

    def __len__(self):
        return len(self.cache)


class BaseDataset(object):
    __ID = 0

    def __init__(
        self, file_map, syncnet_T=5, syncnet_mel_step_size=16,
        face_base_dir='../datasets/extract/mtcnn-wav2lip',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',

        face_path='../stats/all-labels.csv',
        labels_path='../datasets/train.csv',
        detect_path='../stats/mtcnn/labelled-mtcnn.csv',
        log_on_load=False, length=sys.maxsize, face_map=None,
        mel_cache=None
    ):
        super(BaseDataset).__init__()
        print('NEW DATASET')

        if mel_cache is None:
            mel_cache = {}
        else:
            assert isinstance(mel_cache, MelCache)

        # self.ID = self.make_date_stamp()
        self.ID = self.__class__.__ID
        self.__class__.__ID += 1
        self.tag = ''

        self.length = length
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.mel_cache = mel_cache
        self.fps_cache = {}

        self.face_base_dir = face_base_dir
        self.video_base_dir = video_base_dir
        self.audio_base_dir = audio_base_dir

        self.face_path = face_path
        self.labels_path = labels_path
        self.detect_path = detect_path

        self.log_on_load = log_on_load
        self.talker_face_map = None

        if type(file_map) in (str, list, np.ndarray):
            file_map = self.load_folders(file_map, face_map)

        assert type(file_map) is dict
        self.file_map = file_map

    def __len__(self):
        return self.length

    @property
    def num_files(self):
        return len(self.file_map)

    @property
    def name(self):
        return f'{self.tag}{self.ID}'

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M%S")

    def load_talker_face_map(self):
        face_cluster = FaceCluster(load_datasets=False)
        labels_map = face_cluster.get_orig_labels(self.labels_path)

        detections = pd.read_csv(self.detect_path)
        file_column = detections['filename'].to_numpy()
        face_column = detections['face'].to_numpy()
        # frame_column = detections['frame'].to_numpy()
        talker_column = detections['talker'].to_numpy()
        num_faces_column = detections['num_faces'].to_numpy()

        talker_face_map = {}
        filenames = []

        for k in tqdm(range(len(file_column))):
            filename = file_column[k]
            is_fake = labels_map[filename]
            is_talker = talker_column[k]
            num_faces = num_faces_column[k]
            face_no = face_column[k]

            if filename in filenames:
                continue

            if num_faces > 1:
                assert is_talker != -1

                if is_talker == 1:
                    talker_face_map[filename] = face_no
                    filenames.append(filename)
                    continue
            else:
                talker_face_map[filename] = 0
                filenames.append(filename)

        self.talker_face_map = talker_face_map
        return talker_face_map

    def load_folders(self, folders, face_map=None):
        if type(folders) is str:
            basedir = folders
            folders = os.listdir(folders)
        else:
            basedir = self.face_base_dir

        if face_map is None:
            if self.talker_face_map is None:
                self.load_talker_face_map()

            face_map = self.talker_face_map

        file_map = {}
        print(f'FOLDERS FOUND', len(folders))

        for folder in folders:
            if folder.endswith('.mp4'):
                folder = folder[:folder.rindex('.')]

            folder_path = f'{basedir}/{folder}'
            filenames = os.listdir(folder_path)
            video_filename = f'{folder}.mp4'
            if video_filename not in face_map:
                continue

            talker_face_no = face_map[video_filename]
            allowed_image_paths = []

            for filename in filenames:
                face_no, frame_no = self.extract_frame(filename)

                if talker_face_no == -1:
                    assert face_no == 0
                elif talker_face_no != face_no:
                    continue

                if basedir is not None:
                    image_path = f'{basedir}/{folder}/{filename}'
                else:
                    image_path = f'{folder}/{filename}'

                allowed_image_paths.append(image_path)

            file_map[folder] = allowed_image_paths

        return file_map

    def choose_random_name(self):
        names = list(self.file_map.keys())
        name = random.choice(names)
        return name

    def choose_random_filename(self):
        name = self.choose_random_name()
        return f'{name}.mp4'

    def load_image_paths(
        self, name, randomize_images=False,
        num_samples=all
    ):
        if name.endswith('.mp4'):
            name = name[:name.rindex('.mp4')]

        image_paths = self.file_map[name]
        if num_samples is all:
            num_samples = len(image_paths)

        if randomize_images:
            image_paths = random.sample(
                image_paths, k=num_samples
            )

        return image_paths

    def _load_random_video(self, randomize_images=True):
        filename = self.choose_random_filename()
        filename = os.path.basename(filename)

        if self.log_on_load:
            print(f'LOAD VIDEO [{self.name}] {filename}')

        if not filename.endswith('.mp4'):
            assert '.' not in filename
            filename = f'{filename}.mp4'

        name = filename[:filename.rindex('.')]
        image_paths = self.load_image_paths(
            name, randomize_images=randomize_images
        )

        orig_mel = self.load_audio(filename)
        fps = self.resolve_fps(filename)

        try:
            assert fps != 0
        except AssertionError as e:
            print(f'FILE LOAD FAILED', filename)
            raise e

        return orig_mel, image_paths, fps

    def load_audio(self, filename, cache_result=True):
        assert type(filename) is str
        filename = os.path.basename(filename)
        status = f'{len(self.mel_cache)}'
        name = filename

        if self.log_on_load:
            print(f'LOAD AUDIO [{self.name}] {filename} {status}')

        try:
            name = name[:name.rindex('.')]
        except ValueError:
            pass

        if name in self.mel_cache:
            assert type(name) is str
            return self.mel_cache[name]

        assert '.' not in name
        audio_path = f'{self.audio_base_dir}/{name}.flac'
        wav = audio.load_wav(audio_path, hparams.sample_rate)
        orig_mel = audio.melspectrogram(wav).T

        if cache_result:
            self.mel_cache[name] = orig_mel

        return orig_mel

    @staticmethod
    def get_frame_no(filename):
        base_filename = os.path.basename(filename)
        name = base_filename[:base_filename.index('.')]
        face_no, frame_no = [int(x) for x in name.split('-')]
        return frame_no

    @staticmethod
    def extract_frame(filename):
        base_filename = os.path.basename(filename)
        name = base_filename[:base_filename.index('.')]
        face_no, frame_no = [int(x) for x in name.split('-')]
        return face_no, frame_no

    def get_window(self, image_path):
        dirpath = os.path.dirname(image_path)
        face_no, frame_no = self.extract_frame(image_path)

        window_fnames = []
        for frame_id in range(frame_no, frame_no + self.syncnet_T):
            img_filename = f'{face_no}-{frame_id}.jpg'
            frame_path = os.path.join(dirpath, img_filename)
            if not os.path.isfile(frame_path):
                return None

            window_fnames.append(frame_path)

        return window_fnames

    @classmethod
    def batch_image_window(
        cls, images, mirror_prob=0.5, torchify=True
    ):
        window = []
        if random.random() < mirror_prob:
            flip = 1
        else:
            flip = 0

        for image in images:
            image = cls.cv_loader(image, mirror_prob=flip)
            window.append(image)

        batch_x = np.concatenate(window, axis=2) / 255.
        batch_x = batch_x.transpose(2, 0, 1)

        if torchify:
            batch_x = torch.FloatTensor(batch_x)
            batch_x = torch.unsqueeze(batch_x, 0)

        return batch_x

    @classmethod
    def cv_loader(
        cls, img, mirror_prob=0.5, size=None,
        verbose=False
    ):
        if type(img) is str:
            img = cv2.imread(img)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if size is None:
            size = hparams.img_size

        if img.shape[0] != img.shape[1]:
            assert img.shape[0] == img.shape[1] // 2
            img = cv2.resize(img, (size, size // 2))
        else:
            img = cv2.resize(img, (size, size))
            img = img[img.shape[0] // 2:, :]

        if random.random() < mirror_prob:
            cls.m_print(verbose, 'FLIP')
            img = cv2.flip(img, 1)
        else:
            cls.m_print(verbose, 'NO_FLIP')

        return img

    @staticmethod
    def m_print(cond, *args, **kwargs):
        if cond:
            print(*args, **kwargs)

    def load_mel_batch(
        self, orig_mel, fps, frame_no, transpose=False
    ):
        orig_mel = orig_mel.T if transpose else orig_mel
        mel = self.crop_audio_by_frame(
            orig_mel, frame_no=frame_no, fps=fps
        )

        torch_batch_x = torch.FloatTensor(mel.T)
        torch_batch_x = torch_batch_x.unsqueeze(0).unsqueeze(0)
        return torch_batch_x

    def crop_audio_window(self, spec, frame_filename, fps):
        # num frames = len(spec) * fps / 80.0
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_no(frame_filename)
        return self.crop_audio_by_frame(spec, start_frame_num, fps)

    @staticmethod
    def frame_to_index(frame_no, fps):
        return int(80. * (frame_no / float(fps)))

    @staticmethod
    def index_to_frame(index, fps):
        return int(index * (float(fps) / 80.))

    def crop_audio_by_frame(self, spec, frame_no, fps):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = self.frame_to_index(frame_no, fps)
        return self.crop_audio_by_index(spec, start_idx)

    def get_audio_max_frame(self, spec, fps):
        max_index = len(spec) - self.syncnet_mel_step_size
        max_frame = self.index_to_frame(max_index, fps)
        return max_frame

    def crop_audio_by_index(self, spec, start_idx):
        end_idx = start_idx + self.syncnet_mel_step_size
        sub_mel = spec[start_idx: end_idx, :].copy()
        return sub_mel

    def is_complete_mel(self, torch_mel):
        return not self.is_incomplete_mel(torch_mel)

    def is_incomplete_mel(self, torch_mel):
        if self.syncnet_mel_step_size != torch_mel.shape[-1]:
            return True

        return False

    def resolve_fps(self, filename):
        if filename not in self.fps_cache:
            filepath = f'{self.video_base_dir}/{filename}'
            cap = cv2.VideoCapture(filepath)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.fps_cache[filename] = fps

        fps = self.fps_cache[filename]
        return fps

    def resolve_frames(self, filename):
        if filename not in self.frames_cache:
            filepath = f'{self.video_base_dir}/{filename}'
            cap = cv2.VideoCapture(filepath)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames_cache[filename] = frames

        frames = self.frames_cache[filename]
        return frames