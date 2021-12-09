import functools
import time

try:
    import ParentImport
    import audio

    from hparams import hparams, get_image_list
    from ManagerCache import Cache, TrainTypes
    from models import SyncNet_color as SyncNet
    from helpers import Binomial

    from FaceAnalysis import FaceCluster

    from BaseDataset import MelCache
    from BaseDataset import BaseDataset
    from RealDataset import RealDataset
    from FakeDataset import FakeDataset

except ModuleNotFoundError:
    from . import ParentImport
    from . import audio

    from .hparams import hparams, get_image_list
    from .ManagerCache import Cache, TrainTypes
    from .models import SyncNet_color as SyncNet
    from .helpers import Binomial

    from ..FaceAnalysis import FaceCluster

    from .BaseDataset import MelCache
    from .BaseDataset import BaseDataset
    from .RealDataset import RealDataset
    from .FakeDataset import FakeDataset

import torch
import os, random, cv2, argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.utils.data as data_utils
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch import nn
from torch import optim
from glob import glob
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from os.path import dirname, join, basename, isfile
from multiprocessing import Process, Manager, Queue

# syncnet_mel_step_size = 16

def kwargify(**kwargs):
    return kwargs

class SyncDataset(object):
    def __init__(
        self, seed=32, train_size=0.9, load=False,
        mel_step_size=16, syncnet_T=5, train_workers=0,
        test_workers=0, use_joon=False,
        face_base_dir='../datasets/extract/mtcnn-faces',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',
        mel_cache_path=None, transform_image=False
    ):
        assert train_workers % 2 == 0
        assert test_workers % 2 == 0

        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = mel_step_size
        self.train_workers = train_workers
        self.test_workers = test_workers
        self.transform_image = transform_image
        self.use_joon = use_joon

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.h_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.face_base_dir = face_base_dir
        self.video_base_dir = video_base_dir
        self.audio_base_dir = audio_base_dir
        self.mel_cache_path = mel_cache_path

        self.train_size = train_size
        self.seed = seed

        self.fps_cache = {}
        self.frames_cache = {}
        self.mel_cache = None

        self.face_files = None
        self.allowed_filenames = None
        self.train_face_files = None
        self.test_face_files = None
        self.face_map = None

        self.train_real_dataset = None
        self.train_fake_dataset = None
        self.test_real_dataset = None
        self.test_fake_dataset = None
        self.loaded = False

        if load:
            self.load()

    def start_mel_cache(self):
        if self.mel_cache is not None:
            return False

        if (self.train_workers == 0) and (self.test_workers == 0):
            self.mel_cache = MelCache()
            if self.mel_cache_path is not None:
                self.mel_cache.preload(self.mel_cache_path)
        else:
            self.mel_cache = None

        return True

    def _make_base_kwargs(self, log_on_load=False):
        return kwargify(
            log_on_load=log_on_load, mel_cache=self.mel_cache,
            face_map=self.face_map, use_joon=self.use_joon,
            syncnet_mel_step_size=self.syncnet_mel_step_size,
            face_base_dir=self.face_base_dir,
            video_base_dir=self.video_base_dir,
            audio_base_dir=self.audio_base_dir,
            transform_image=self.transform_image
        )

    def make_data_loader(self, *args, **kwargs):
        dataset = self.make_dataset(*args, **kwargs)
        dataloader = data_utils.DataLoader(
            dataset, batch_size=None, num_workers=0,
            pin_memory=False
        )

        return dataset, dataloader

    def make_dataset(
        self, file_map=None, mel_cache=True, **kwargs
    ):
        if file_map is None:
            file_map = self.train_face_files

        if mel_cache is True:
            self.start_mel_cache()
        elif mel_cache is not None:
            assert isinstance(mel_cache, MelCache)
            self.mel_cache = mel_cache

        base_kwargs = self._make_base_kwargs()
        return RealDataset(
            file_map, **base_kwargs, **kwargs
        )

    def start_data_loaders(
        self, log_on_load=True, image_cache_workers=0,
        start_train_workers=True, start_test_workers=True
    ):
        self.start_mel_cache()
        base_kwargs = self._make_base_kwargs(log_on_load)

        real_wrap = functools.partial(RealDataset, **base_kwargs)
        fake_wrap = functools.partial(FakeDataset, **base_kwargs)
        start_train_workers &= (image_cache_workers > 0)
        start_test_workers &= (image_cache_workers > 0)

        train_kwargs = kwargify(
            file_map=self.train_face_files,
            num_image_cache_processes=image_cache_workers,
            start_mp_image_cache=start_train_workers
        )
        test_kwargs = kwargify(
            file_map=self.test_face_files,
            num_image_cache_processes=image_cache_workers,
            start_mp_image_cache=start_test_workers
        )

        self.train_real_dataset = real_wrap(**train_kwargs)
        self.train_fake_dataset = fake_wrap(**train_kwargs)
        self.test_real_dataset = real_wrap(**test_kwargs)
        self.test_fake_dataset = fake_wrap(**test_kwargs)

        print(f'TRAIN REAL LOAD', self.train_real_dataset.num_files)
        print(f'TRAIN FAKE LOAD', self.train_fake_dataset.num_files)
        print(f'TEST REAL LOAD', self.test_real_dataset.num_files)
        print(f'TEST FAKE LOAD', self.test_fake_dataset.num_files)

    def load(self, *args, **kwargs):
        self.load_datasets()
        self.start_data_loaders(*args, **kwargs)
        self.loaded = True

    def load_datasets(
        self, exclude_fakes=True, exclude_multiple_faces=False
    ):
        face_cluster = FaceCluster(load_datasets=False)
        face_path = '../stats/all-labels.csv'
        face_map = face_cluster.make_face_map(face_path)
        labels_path = '../datasets/train.csv'
        labels_map = face_cluster.get_orig_labels(labels_path)
        detect_path = '../stats/mtcnn/labelled-mtcnn.csv'
        detections = pd.read_csv(detect_path)

        self.face_files = {}
        self.train_face_files = {}
        self.test_face_files = {}
        self.face_map = {}

        # print(detections)
        file_column = detections['filename'].to_numpy()
        num_face_column = detections['num_faces'].to_numpy()
        face_no_column = detections['face'].to_numpy()
        talker_column = detections['talker'].to_numpy()
        unique_filenames = np.unique(file_column)
        allowed_filenames = []

        fakes_excluded = 0
        multi_faces_excluded = 0
        excluded_filenames = []
        face_map = {}

        for k, filename in enumerate(tqdm(file_column)):
            is_fake = labels_map[filename]
            num_faces = num_face_column[k]
            face_no = face_no_column[k]
            talker = talker_column[k]

            if talker == -1:
                assert num_faces == 1
                face_map[filename] = 0
            elif talker == 1:
                face_map[filename] = face_no

            if filename in allowed_filenames:
                continue
            if filename in excluded_filenames:
                continue

            if exclude_fakes and is_fake:
                excluded_filenames.append(filename)
                fakes_excluded += 1
                continue

            if num_faces > 1:
                if exclude_multiple_faces:
                    excluded_filenames.append(filename)
                    multi_faces_excluded += 1
                    continue

                assert talker != -1
                assert num_faces == 2

            allowed_filenames.append(filename)

        train_files, test_files, _, _ = train_test_split(
            allowed_filenames, allowed_filenames,
            random_state=self.seed, train_size=self.train_size
        )

        print(f'FAKES EXCLUDED', fakes_excluded)
        print(f'MULTI-FACES EXCLUDED', multi_faces_excluded)
        print(f'TRAIN FILES: {len(train_files)}')
        print(f'TEST FILES: {len(test_files)}')

        self.face_files = unique_filenames
        self.allowed_filenames = allowed_filenames
        self.train_face_files = train_files
        self.test_face_files = test_files
        self.face_map = face_map

    def prepare_batch(
        self, batch_size, fake_p=0.5, is_training=True,
        randomize=True, filename=None, binomial_sampling=False
    ):
        if binomial_sampling:
            fake_count = Binomial.random_sample(
                batch_size, prob=fake_p
            )
        else:
            fake_count = int(batch_size * fake_p)

        real_count = batch_size - fake_count
        real_images, real_mels = self.load_samples(
            0, real_count, is_training=is_training,
            filename=filename
        )
        fake_images, fake_mels = self.load_samples(
            1, fake_count, is_training=is_training,
            filename=filename
        )

        mels = real_mels + fake_mels
        images = real_images + fake_images
        labels = [0] * real_count + [1] * fake_count

        if randomize:
            indexes = np.arange(batch_size)
            np.random.shuffle(indexes)

            images = self.reorder(images, indexes)
            labels = self.reorder(labels, indexes)
            mels = self.reorder(mels, indexes)

        return labels, images, mels

    @staticmethod
    def reorder(items, indexes):
        new_items = []
        for index in indexes:
            new_items.append(items[index])

        return new_items

    def load_samples(
        self, label, num_samples, filename=None,
        is_training=True
    ):
        mel_samples, img_samples = [], []

        for k in range(num_samples):
            sample = self._load_sample(
                label, is_training=is_training,
                filename=filename
            )
            torch_img, torch_mel = sample
            img_samples.append(torch_img)
            mel_samples.append(torch_mel)

        return img_samples, mel_samples

    def _load_sample(self, label, filename=None, is_training=True):
        assert label in (0, 1)
        assert type(is_training) is bool
        assert self.loaded

        if is_training:
            if label == 0:
                dataset = self.train_real_dataset
            else:
                dataset = self.train_fake_dataset
        else:
            if label == 0:
                dataset = self.test_real_dataset
            else:
                dataset = self.test_fake_dataset

    def start_processes(self):
        self.manager = Manager()
        self.cache = Cache(self.manager)

        for k in range(self.num_audio_workers):
            audio_process = Process(target=self.build_fake_audios)
            self.audio_processes.append(audio_process)
            audio_process.daemon = True
            audio_process.start()

        assert dataset is not None
        if isinstance(dataset, DataLoader):
            dataset = dataset.dataset

        sample = dataset.choose_sample(filename=filename)
        torch_img, torch_mel = sample
        return torch_img, torch_mel

    @staticmethod
    def torchify_batch(labels, images, mels):
        try:
            torch_mels = torch.cat(mels, dim=0)
            torch_images = torch.cat(images, dim=0)
            torch_labels = torch.FloatTensor(labels)
        except RuntimeError as e:
            print('TORCH FAIL')
            raise e

        # torch_labels = torch.unsqueeze(torch_labels, 1)
        return torch_labels, torch_images, torch_mels

    @staticmethod
    def pil_loader(path: str, mirror_prob=0.5) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.resize(
                (hparams.img_size, hparams.img_size),
                Image.ANTIALIAS
            )

            if random.random() < mirror_prob:
                img = ImageOps.mirror(img)

            return img.convert('RGB')

    @staticmethod
    def cv_loader(*args, **kwargs):
        return BaseDataset._cv_loader(*args, **kwargs)

    @staticmethod
    def load_cct(*args, **kwargs):
        return BaseDataset.load_cct(*args, **kwargs)

    @staticmethod
    def m_print(cond, *args, **kwargs):
        if cond:
            print(*args, **kwargs)

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

    @staticmethod
    def batch_images_joon(
        images, mirror_prob=0.5, torchify=True, size=224,
        transform_image=None
    ):
        return BaseDataset.batch_image_window_joon(
            images, mirror_prob=mirror_prob, torchify=torchify,
            size=size, transform_image=transform_image
        )

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

    def build_fake_audios(self):
        cache = self.cache

        while not self.kill:
            if cache.min_mel_size > self.max_cache_size:
                # print(f'TOO FAT', cache.min_mel_size)
                time.sleep(1)
                continue

            train_fakes = cache.train_fake_mel_size
            test_fakes = cache.test_fake_mel_size
            # print('TFF', train_fakes, test_fakes)

            if train_fakes < test_fakes:
                self.load_fake_audios(is_training=True)
            else:
                self.load_fake_audios(is_training=False)

    def load_fake_audios(self, is_training=True):
        if is_training:
            filenames = self.train_face_files.keys()
        else:
            filenames = self.test_face_files.keys()

        filenames = list(filenames)
        filename = random.choice(filenames)
        orig_mel = self.load_audio(filename)
        fps = self.resolve_fps(filename)

        num_frames = len(orig_mel) * fps / 80.0
        max_start_index = len(orig_mel) - self.syncnet_mel_step_size
        # print('O-MEL SHAPE', orig_mel.shape)

        assert max_start_index > 0
        num_samples = int(num_frames / self.sample_framerate)
        indexes = random.sample(range(max_start_index), k=num_samples)

        for start_index in indexes:
            sub_mel = self.crop_audio_by_index(orig_mel, start_index)
            torch_mels = torch.FloatTensor(sub_mel.T)
            torch_mels = torch_mels.unsqueeze(0).unsqueeze(0)
            # print('MEL-LEN', torch_mels.shape)
            assert not self.is_incomplete_mel(torch_mels)

            if is_training:
                self.cache.add_train_fake_mel(torch_mels)
            else:
                self.cache.add_test_fake_mel(torch_mels)

    def load_audio(self, filename):
        filename = basename(filename)
        name = filename

        try:
            name = name[:name.index('.')]
        except IndexError:
            pass

        audio_path = f'{self.audio_base_dir}/{name}.flac'
        wav = audio.load_wav(audio_path, hparams.sample_rate)
        orig_mel = audio.melspectrogram(wav).T
        return orig_mel

    @staticmethod
    def filter_image_paths(image_paths):
        filtered_paths = []
        for image_path in image_paths:
            if image_path.endswith('0.jpg'):
                filtered_paths.append(image_path)

        return filtered_paths

    @staticmethod
    def get_frame_no(filename):
        base_filename = basename(filename)
        name = base_filename[:base_filename.index('.')]
        face_no, frame_no = [int(x) for x in name.split('-')]
        return frame_no

    @staticmethod
    def extract_frame(filename):
        base_filename = basename(filename)
        name = base_filename[:base_filename.index('.')]
        face_no, frame_no = [int(x) for x in name.split('-')]
        return face_no, frame_no

    def get_window(self, image_path):
        dirpath = os.path.dirname(image_path)
        face_no, frame_no = self.extract_frame(image_path)

        window_fnames = []
        for frame_id in range(frame_no, frame_no + self.syncnet_T):
            img_filename = f'{face_no}-{frame_id}.jpg'
            frame_path = join(dirpath, img_filename)
            if not isfile(frame_path):
                return None

            window_fnames.append(frame_path)

        return window_fnames

    def _get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vid_name = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vid_name, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None

            window_fnames.append(frame)

        return window_fnames

    def is_incomplete_mel(
        self, torch_mel, syncnet_mel_step_size=None
    ):
        if syncnet_mel_step_size is None:
            syncnet_mel_step_size = self.syncnet_mel_step_size

        if syncnet_mel_step_size != torch_mel.shape[-1]:
            return True

        return False

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

    def load_mel_joon(self, *args, **kwargs):
        return self.load_mel_batch_joon(*args, **kwargs)

    def load_mel_batch_joon(
        self, cct, fps, frame_no, syncnet_mel_step_size=None
    ):
        if syncnet_mel_step_size is None:
            syncnet_mel_step_size = self.syncnet_mel_step_size

        start_idx = int(100. * (frame_no / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        mel = cct[:, :, :, start_idx: end_idx]

        torch_batch_x = torch.FloatTensor(mel)
        return torch_batch_x

    def crop_audio_by_frame(self, spec, frame_no, fps):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = int(80. * (frame_no / float(fps)))
        return self.crop_audio_by_index(spec, start_idx)

    def crop_audio_by_index(self, spec, start_idx):
        end_idx = start_idx + self.syncnet_mel_step_size
        return spec[start_idx: end_idx, :]

    def crop_audio_window(self, spec, frame_filename, fps):
        # num frames = len(spec) * fps / 80.0
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_no(frame_filename)
        return self.crop_audio_by_frame(spec, start_frame_num, fps)
