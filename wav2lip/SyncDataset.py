import time

try:
    import ParentImport

    from FaceAnalysis import FaceCluster

except ModuleNotFoundError:
    from . import ParentImport

    from ..FaceAnalysis import FaceCluster

import audio
import torch
import os, random, cv2, argparse
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np

from ManagerCache import Cache, TrainTypes

from tqdm import tqdm
from torch import nn
from torch import optim
from glob import glob
from PIL import Image, ImageOps
from torchvision import transforms
from hparams import hparams, get_image_list
from torch.utils import data as data_utils
from models import SyncNet_color as SyncNet
from sklearn.model_selection import train_test_split
from os.path import dirname, join, basename, isfile
from multiprocessing import Process, Manager, Queue

# syncnet_mel_step_size = 16

def kwargify(**kwargs):
    return kwargs

class SyncDataset(object):
    def __init__(
        self, seed=32, train_size=0.95, load=True,
        mel_step_size=16
    ):
        self.mel_step_size = mel_step_size
        self.syncnet_mel_step_size = 16
        self.syncnet_T = 1

        self.sample_framerate = 20
        self.max_cache_size = 1000
        self.min_samples = 200

        self.num_audio_workers = 2
        self.num_build_workers = 2
        self.audio_processes = []
        self.build_processes = []

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

        self.face_base_dir = '../datasets-local/mtcnn-faces'
        self.video_base_dir = '../datasets/train/videos'
        self.audio_base_dir = '../datasets-local/audios-flac'

        self.face_files = None
        self.train_face_files = None
        self.test_face_files = None

        self.train_size = train_size
        self.seed = seed

        self.fps_cache = {}
        self.frames_cache = {}

        self.manager = Manager()
        self.cache = Cache(self.manager)
        self.train_fake_audio_queue = Queue()
        self.test_fake_audio_queue = Queue()
        self.kill = False

        if load:
            self.load_datasets()
            self.start_processes()

    def load_datasets(self):
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

        file_column = detections['filename'].to_numpy()
        face_column = detections['face'].to_numpy()
        frame_column = detections['frame'].to_numpy()
        talker_column = detections['talker'].to_numpy()
        num_faces_column = detections['num_faces'].to_numpy()
        file_map = {}

        for k in tqdm(range(len(file_column))):
            filename = file_column[k]
            is_fake = labels_map[filename]
            is_talker = talker_column[k]
            num_faces = num_faces_column[k]

            if is_fake:
                continue
            elif not is_talker and (num_faces > 1):
                continue

            if filename not in file_map:
                file_map[filename] = []

            frames = file_map[filename]
            frames.append({
                'face': face_column[k],
                'frame': frame_column[k],
                'talker': is_talker,
                'num_faces': num_faces
            })

        for filename in tqdm(face_map):
            if filename not in file_map:
                continue

            # face_fake = face_map[filename]
            frames = file_map[filename]
            self.face_files[filename] = []

            for k in range(len(frames)):
                frame_no = frames[k]['frame']
                # is_talker = frames[k]['talker']
                face_no = frames[k]['face']

                name = filename[:filename.index('.')]
                img_file = f'{name}/{face_no}-{frame_no}.jpg'
                img_path = f'{self.face_base_dir}/{img_file}'
                self.face_files[filename].append(img_path)

        all_filenames = list(self.face_files.keys())
        x_train, x_test, y_train, y_test = train_test_split(
            all_filenames, all_filenames,
            random_state=self.seed, train_size=self.train_size
        )

        self.train_face_files = {}
        self.test_face_files = {}

        for filename in all_filenames:
            video_face_files = self.face_files[filename]
            if filename in x_train:
                self.train_face_files[filename] = video_face_files
            else:
                self.test_face_files[filename] = video_face_files

    def prepare_batch(
        self, batch_size, fake_p=0.5,
        is_training=True, randomize=True
    ):
        fake_count = int(batch_size * fake_p)
        real_count = batch_size - fake_count

        real_images, real_mels = self.safe_load_samples(
            0, real_count, is_training=is_training
        )
        fake_images, fake_mels = self.safe_load_samples(
            1, fake_count, is_training=is_training
        )

        mels = real_mels + fake_mels
        images = real_images + fake_images
        labels = [0] * real_count + [1] * fake_count
        return labels, images, mels

    @staticmethod
    def torch_batch(labels, images, mels):
        torch_images = torch.cat(images)
        torch_images = torch.unsqueeze(torch_images, 0)
        """
        torch_images = torch.cat([
            torch.unsqueeze(image, 0) for image in images
        ])
        torch_mels = torch.cat([
            torch.unsqueeze(torch.FloatTensor(mel), 0)
            for mel in mels
        ])
        """
        print('NN', mels[0].shape, mels[0].T.shape)
        print([mel.T.shape for mel in mels])

        torch_mels = torch.cat([
            torch.unsqueeze(torch.FloatTensor(mel.T), 0)
            for mel in mels
        ])
        torch_mels = torch.unsqueeze(torch_mels, -1)
        torch_mels = torch_mels.permute(0, 3, 1, 2)

        torch_labels = torch.FloatTensor(labels)
        return torch_labels, torch_images, torch_mels

    def safe_load_samples(self, label, num_samples, is_training=True):
        assert label in (0, 1)

        turns = 0
        while self.cache.min_samples < self.min_samples:
            sizes = self.get_sample_sizes()
            print(f'WAITING [{turns}] {sizes}')
            time.sleep(1)
            turns += 1

        return self.cache.pop_samples(
            label, num_samples, is_training=is_training
        )

    def start_processes(self):
        for k in range(self.num_audio_workers):
            audio_process = Process(target=self.build_fake_audios)
            self.audio_processes.append(audio_process)
            audio_process.daemon = True
            audio_process.start()

        for k in range(self.num_build_workers):
            build_process = Process(target=self.build_sample_loop)
            self.build_processes.append(build_process)
            build_process.daemon = True
            build_process.start()

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
        assert max_start_index > 0
        num_samples = int(num_frames / self.sample_framerate)
        indexes = random.sample(range(max_start_index), k=num_samples)

        for start_index in indexes:
            end_index = start_index + self.syncnet_mel_step_size
            mel = orig_mel[start_index: end_index, :]
            assert mel.shape[0] == self.syncnet_mel_step_size

            if is_training:
                self.cache.add_train_fake_mel(mel)
            else:
                self.cache.add_test_fake_mel(mel)

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

    def load_real_samples(
        self, filename=None, is_training=True
    ):
        if is_training:
            file_map = self.train_face_files
            train_type = TrainTypes.TRAIN_REAL
        else:
            file_map = self.test_face_files
            train_type = TrainTypes.TEST_REAL

        if filename is None:
            filename = random.choice(list(
                file_map.keys()
            ))

        image_paths = file_map[filename]
        fps = self.resolve_fps(filename)
        orig_mel = self.load_audio(filename)

        for image_path in image_paths:
            frame_no = self.get_frame_no(image_path)
            # print(f'FRAME NO', frame_no)

            image = self.pil_loader(image_path)
            image = self.transform(image)
            bottom_img = image[:, image.shape[1]//2:]

            frame_key = (filename, frame_no)
            mel = self.crop_audio_by_frame(orig_mel, frame_no, fps)
            self.cache.add(train_type, frame_key, bottom_img, mel)

    def load_fake_samples(
        self, filename=None, is_training=True
    ):
        if is_training:
            file_map = self.train_face_files
            train_type = TrainTypes.TRAIN_FAKE
        else:
            file_map = self.test_face_files
            train_type = TrainTypes.TEST_FAKE

        if filename is None:
            filenames = list(file_map.keys())
            filename = random.choice(filenames)

        image_paths = file_map[filename]
        num_frames = self.resolve_frames(filename)
        num_samples = int(num_frames / sample_framerate)
        num_samples = min(len(image_paths), num_samples)
        sample_paths = random.sample(
            image_paths, k=num_samples
        )

        for image_path in sample_paths:
            frame_no = self.get_frame_no(image_path)
            frame_key = (filename, frame_no)

            window_fnames = self.get_window(image_path)
            if window_fnames is None:
                continue

            image_window = self.load_image_window(window_fnames)
            # image = self.pil_loader(image_path)
            # image = self.transform(image)
            # bottom_img = image[:, image.shape[1]//2:]

            mel = self.cache.safe_pop_fake_mel(is_training)
            self.cache.add(train_type, frame_key, image_window, mel)

    def build_sample_loop(self):
        while not self.kill:
            self.build_samples()

    def build_samples(self):
        T = TrainTypes

        if self.cache.get_size(T.TRAIN_REAL) < self.max_cache_size:
            # print('LOAD REAL-TRAIN')
            self.load_real_samples(is_training=True)
        if self.cache.get_size(T.TEST_REAL) < self.max_cache_size:
            # print('LOAD REAL-TEST')
            self.load_real_samples(is_training=False)

        if self.cache.get_size(T.TRAIN_FAKE) < self.max_cache_size:
            # print('LOAD FAKE-TRAIN')
            self.load_fake_samples(is_training=True)
        if self.cache.get_size(T.TEST_FAKE) < self.max_cache_size:
            # print('LOAD FAKE-TRAIN')
            self.load_fake_samples(is_training=False)

    def get_sample_sizes(self):
        return (
            self.cache.get_size(TrainTypes.TRAIN_REAL),
            self.cache.get_size(TrainTypes.TRAIN_FAKE),
            self.cache.get_size(TrainTypes.TEST_REAL),
            self.cache.get_size(TrainTypes.TEST_FAKE)
        )

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

    @staticmethod
    def load_image_window(window_fnames, mirror_prob=0.5):
        # need to implement flipping
        size = hparams.size
        image_window = []

        for item in window_fnames:
            if type(item) is str:
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = item

            if img is None:
                return False

            if img.shape[0] != img.shape[1]:
                assert img.shape[0] == img.shape[1] // 2
                img = cv2.resize(img, (size, size // 2))
            else:
                img = cv2.resize(img, (size, size))
                img = img[image.shape[0] // 2:, :]

            assert img.shape[0] == img.shape[1] // 2
            image_window.append(img)

        return image_window

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

    def crop_audio_by_frame(self, spec, frame_no, fps):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_idx = int(80. * (frame_no / float(fps)))
        end_idx = start_idx + self.syncnet_mel_step_size
        return spec[start_idx: end_idx, :]

    def crop_audio_window(self, spec, start_frame, fps):
        # num frames = len(spec) * fps / 80.0
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_no(start_frame)
        return self.crop_audio_by_frame(spec, start_frame_num, fps)
