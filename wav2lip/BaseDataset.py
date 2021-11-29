from abc import ABC
from typing import Iterator

try:
    import ParentImport
    import audio

    from FaceAnalysis import FaceCluster
    from FaceImageMap import FaceImageMap
    from DeepfakeDetection.FaceExtractor import FaceImage
    from hparams import hparams
except ModuleNotFoundError:
    from . import ParentImport
    from . import audio

    from ..FaceAnalysis import FaceCluster
    from ..FaceImageMap import FaceImageMap
    from .DeepfakeDetection.FaceExtractor import FaceImage
    from .hparams import hparams

import os
import sys
import random
import pandas as pd
import python_speech_features
import numpy as np
import torch
import time
import cv2
import re

from tqdm.auto import tqdm
from torch.utils.data import IterableDataset
from datetime import datetime as Datetime
from torch.utils import data as data_utils
from torchvision import transforms

from multiprocessing import Process, Manager


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

    def preload(self, mel_cache_path, use_joon=True):
        print(f'LOADING MEL CACHE: {mel_cache_path}')

        file_obj = open(mel_cache_path, 'rb')
        preprocessed = np.load(file_obj, allow_pickle=True)
        self.cache = preprocessed[()]

        if use_joon and (self.cache['cache_type'] != 'cct'):
            raise ValueError('NOT USING JOON')

class BaseDataset(object):
    __ID = 0
    transform_image = transforms.Normalize(
        [0.5] * 3, [0.5] * 3
    )

    def __init__(
        self, file_map, syncnet_T=5, syncnet_mel_step_size=16,
        face_base_dir='../datasets/extract/mtcnn-wav2lip',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',

        face_path='../stats/all-labels.csv',
        labels_path='../datasets/train.csv',
        detect_path='../stats/mtcnn/labelled-mtcnn.csv',

        log_on_load=False, length=sys.maxsize, face_map=None,
        mel_cache=None, use_joon=False, transform_image=False,
        mp_image_cache_size=0, start_mp_image_cache=False,
        num_image_cache_processes=1
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
        self.torch_img_cache = None
        self.mel_cache = mel_cache
        self.fps_cache = {}

        self.mp_image_cache_size = mp_image_cache_size
        self.mp_image_cache_started = False
        self.num_image_cache_processes = num_image_cache_processes
        self.mp_image_cache_processes = None

        self.face_base_dir = face_base_dir
        self.video_base_dir = video_base_dir
        self.audio_base_dir = audio_base_dir

        self.face_path = face_path
        self.labels_path = labels_path
        self.detect_path = detect_path

        self.use_joon = use_joon
        self.log_on_load = log_on_load
        self.transform_image = transform_image
        self.talker_face_map = None

        if type(file_map) in (str, list, np.ndarray):
            file_map = self.load_folders(file_map, face_map)

        assert type(file_map) is dict
        self.file_map = file_map

        if start_mp_image_cache:
            self.start_img_cacher()

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

        for folder in tqdm(folders):
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

    @staticmethod
    def kwargify(**kwargs):
        return kwargs

    def start_img_cacher(self):
        assert self.mp_image_cache_size > 0
        assert not self.mp_image_cache_started
        assert self.mp_image_cache_processes is None

        self.mp_image_cache_started = True
        self.mp_image_cache_processes = []

        manager = Manager()
        self.torch_img_cache = manager.list()

        for process in range(self.num_image_cache_processes):
            process = Process(target=self._build_torch_img_cache)
            self.mp_image_cache_processes.append(process)
            process.daemon = True
            process.start()

    def _build_torch_img_cache(self, mirror_prob=0.5):
        assert self.mp_image_cache_size > 0

        while True:
            image_path, torch_imgs = self.load_torch_window(
                filename, frame_no=None, face_no=None,
                mirror_prob=mirror_prob
            )

            image_sample = (image_path, torch_imgs)
            if len(self.torch_img_cache) < self.mp_image_cache_size:
                self.torch_img_cache.append(image_sample)

    def fetch_torch_img_sample(self, blocking=True):
        assert self.mp_image_cache_size > 0

        while True:
            try:
                sample = self.torch_img_cache.pop()
                return sample
            except IndexError as e:
                if not blocking:
                    raise e

            time.sleep(0.1)

    def choose_sample(self, *args, **kwargs):
        raise NotImplemented

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

    def load_audio(
        self, filename, cache_result=True, sample_rate=None
    ):
        if sample_rate is None:
            sample_rate = hparams.sample_rate

        assert type(filename) is str
        filename = os.path.basename(filename)
        status = f'{len(self.mel_cache)}'
        name = filename

        try:
            name = name[:name.rindex('.')]
        except ValueError:
            pass

        if name in self.mel_cache:
            assert type(name) is str
            return self.mel_cache[name]

        if self.log_on_load:
            print(f'LOAD AUDIO [{self.name}] {filename} {status}')

        orig_mel = self.load_audio_file(
            name, audio_base_dir=self.audio_base_dir,
            sample_rate=sample_rate, use_joon=self.use_joon
        )

        if cache_result:
            self.mel_cache[name] = orig_mel

        return orig_mel

    @classmethod
    def load_audio_file(
        cls, name, audio_base_dir, sample_rate=None,
        use_joon=True
    ):
        if sample_rate is None:
            sample_rate = hparams.sample_rate

        assert '.' not in name
        audio_path = f'{audio_base_dir}/{name}.flac'
        wav = audio.load_wav(audio_path, sample_rate)

        if not use_joon:
            orig_mel = audio.melspectrogram(wav).T
        else:
            orig_mel = cls.load_cct(
                wav, sample_rate=sample_rate,
                torchify=False
            )

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

    @classmethod
    def filter_image_paths(
        cls, image_paths, target_frame_no=None, target_face_no=None,
        ensure_single=False
    ):
        target_image_paths = []

        for image_path in image_paths:
            face_no, frame_no = cls.extract_frame(image_path)

            if frame_no != target_frame_no:
                if target_face_no is not None:
                    continue
            elif face_no != target_face_no:
                if target_face_no is not None:
                    continue

            target_image_paths.append(image_path)

        assert not ensure_single or (len(target_image_paths) == 1)
        return target_image_paths

    def load_torch_window(
        self, filename, frame_no=None, face_no=None,
        mirror_prob=0.5
    ):
        window_fnames, image_path = None, None

        while window_fnames is None:
            image_paths = self.load_image_paths(
                filename, randomize_images=True,
                num_samples=all
            )
            image_path = self.filter_image_paths(
                image_paths, target_frame_no=frame_no,
                target_face_no=face_no, ensure_single=False
            )[0]

            window_fnames = self.get_window(image_path)

        torch_imgs = self.batch_image_window(
            window_fnames, mirror_prob=mirror_prob
        )

        return image_path, torch_imgs

    def load_torch_images(self, filename=None):
        if self.mp_image_cache_started and (filename is None):
            sample = self.fetch_torch_img_sample()
            image_path, torch_imgs = sample

            name = os.path.basename(image_path)
            filename = f"{name[:name.rindex('.')]}.mp4"
            return filename, image_path, torch_imgs

        if filename is None:
            name = self.choose_random_name()
            filename = f'{name}.mp4'

        assert type(filename) is str
        current_fps = self.resolve_fps(filename)
        assert current_fps != 0

        image_path, torch_imgs = self.load_torch_window(filename)
        return filename, image_path, torch_imgs

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

    def batch_image_window(self, *args, **kwargs):
        assert self.use_joon
        return self.batch_image_window_joon(
            *args, **kwargs
        )
    
    @classmethod
    def batch_image_window_joon(
        cls, images, mirror_prob=0.5, size=224, torchify=True,
        transform_image=None
    ):
        if transform_image is None:
            transform_image = cls.transform_image

        window = []
        if random.random() < mirror_prob:
            flip = 1
        else:
            flip = 0

        for image in images:
            try:
                image = cls._cv_loader(
                    image, mirror_prob=flip, size=size,
                    assert_square=True, bottom_half=False
                )
            except AssertionError as e:
                print('CV LOAD FAILED')
                raise e

            t_image = torch.FloatTensor(image)
            t_image = torch.permute(t_image, (2, 0, 1))
            if transform_image:
                t_image = transform_image(t_image)

            t_image = t_image.unsqueeze(0)
            window.append(t_image)

        im_batch = torch.stack(window, dim=2)
        if not torchify:
            im_batch = im_batch.numpy()

        return im_batch

    @classmethod
    def batch_images_wav2lip(cls, *args, **kwargs):
        return self.batch_image_window_wav2lip(*args, **kwargs)

    @classmethod
    def batch_image_window_wav2lip(
        cls, images, mirror_prob=0.5, torchify=True, size=None
    ):
        window = []
        if random.random() < mirror_prob:
            flip = 1
        else:
            flip = 0

        for image in images:
            image = cls.load_image(image, mirror_prob=flip, size=size)
            window.append(image)

        batch_x = np.concatenate(window, axis=2) / 255.
        batch_x = batch_x.transpose(2, 0, 1)

        if torchify:
            batch_x = torch.FloatTensor(batch_x)
            batch_x = torch.unsqueeze(batch_x, 0)

        return batch_x

    @classmethod
    def _cv_loader(
        cls, img, mirror_prob=0.5, size=True,
        verbose=False, bottom_half=True, assert_square=True
    ):
        if size is True:
            size = hparams.img_size

        return cls.load_image(
            img, mirror_prob=mirror_prob, size=size,
            verbose=verbose, bottom_half=bottom_half,
            assert_square=assert_square
        )

    @classmethod
    def load_image(
        cls, img, size, mirror_prob=0,
        verbose=False, bottom_half=False, assert_square=True
    ):
        if type(img) is str:
            img = cv2.imread(img)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = img.shape[0], img.shape[1]
        resize_width, resize_height = size, size

        if bottom_half:
            if height != width:
                assert height == width // 2
                resize_height = size // 2
            else:
                assert not assert_square or (width == height)
                img = img[height // 2:, :]
        else:
            assert not assert_square or (width == height)

        if size is not None:
            img = cv2.resize(
                img, (resize_width, resize_height)
            )

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

    @staticmethod
    def load_cct(
        raw_audio, sample_rate=None, torchify=True
    ):
        if sample_rate is None:
            sample_rate = hparams.sample_rate

        mfcc = zip(*python_speech_features.mfcc(
            raw_audio, sample_rate
        ))

        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)

        if torchify:
            cct = torch.autograd.Variable(
                torch.from_numpy(cc.astype(float)).float()
            )
            return cct
        else:
            return cc

    def load_mel_batch(self, *args, **kwargs):
        if not self.use_joon:
            return self.load_mel_batch_wav2lip(*args, **kwargs)
        else:
            return self._load_mel_batch_joon(*args, **kwargs)

    def _load_mel_batch_joon(
        self, *args, syncnet_mel_step_size=None, **kwargs
    ):
        if syncnet_mel_step_size is None:
            syncnet_mel_step_size = self.syncnet_mel_step_size

        return self.load_mel_batch_joon(
            *args, syncnet_mel_step_size=syncnet_mel_step_size,
            **kwargs
        )

    @staticmethod
    def load_mel_batch_joon(
        cct, fps, frame_no, syncnet_mel_step_size
    ):
        start_idx = int(100. * (frame_no / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        mel = cct[:, :, :, start_idx: end_idx]

        torch_batch_x = torch.FloatTensor(mel)
        return torch_batch_x

    def load_mel_batch_wav2lip(
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
    def frame_to_index(frame_no, fps, mels_per_second=80.0):
        return int(mels_per_second * (frame_no / float(fps)))

    @staticmethod
    def index_to_frame(index, fps, mels_per_second=80.0):
        return int(index * (float(fps) / mels_per_second))

    def crop_audio_by_frame(self, spec, frame_no, fps):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = self.frame_to_index(frame_no, fps)
        return self.crop_audio_by_index(spec, start_idx)

    def get_audio_max_frame(self, spec, fps):
        if self.use_joon:
            length = spec.shape[-1]
            mels_per_second = 100.
        else:
            length = spec.shape[0]
            mels_per_second = 80.

        max_index = length - self.syncnet_mel_step_size
        max_frame = self.index_to_frame(
            max_index, fps, mels_per_second=mels_per_second
        )
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

    @staticmethod
    def load_fps(filename, video_base_dir):
        filepath = f'{video_base_dir}/{filename}'
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return fps

    def resolve_fps(self, filename):
        if filename not in self.fps_cache:
            fps = self.load_fps(filename, self.video_base_dir)
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

    def get_video_image_paths(self, filename):
        filename = os.path.basename(filename)
        folder = filename[:filename.rindex('.')]
        image_paths = self.file_map[folder]
        return image_paths

    def load_face_image_map(self, filename, img_size=224):
        filename = os.path.basename(filename)
        fps = self.resolve_fps(filename)
        folder = filename[:filename.rindex('.')]
        face_image_map = {}

        face_nos = []
        image_paths = self.file_map[folder]

        for image_path in image_paths:
            face_no, frame_no = self.extract_frame(image_path)
            if face_no not in face_nos:
                face_nos.append(face_no)

        num_faces = len(face_nos)

        for image_path in image_paths:
            face_no, frame_no = self.extract_frame(image_path)
            detected = frame_no % 10 == 0
            np_image = self.load_image(
                image_path, assert_square=True, bottom_half=False,
                mirror_prob=0, size=img_size
            )

            if face_no not in face_image_map:
                face_image_map[face_no] = {}

            face_image = FaceImage(
                image=np_image, coords=None, strict=False,
                face_no=face_no, frame_no=frame_no,
                num_faces=num_faces, detected=detected
            )

            face_images = face_image_map[face_no]
            face_images[frame_no] = face_image

        face_image_map = FaceImageMap(face_image_map, fps=fps)
        return face_image_map, num_faces