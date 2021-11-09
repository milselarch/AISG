try:
    import ParentImport

    from FaceAnalysis import FaceCluster
except ModuleNotFoundError:
    from . import ParentImport

    from ..FaceAnalysis import FaceCluster

import os
import audio
import random
import pandas as pd
import numpy as np
import torch
import cv2

from tqdm.auto import tqdm
from hparams import hparams

class BaseDataset(object):
    def __init__(
        self, file_map, syncnet_T=5, syncnet_mel_step_size=16,
        face_base_dir='../datasets-local/mtcnn-faces',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets-local/audios-flac',

        face_path='../stats/all-labels.csv',
        labels_path='../datasets/train.csv',
        detect_path='../stats/mtcnn/labelled-mtcnn.csv'
    ):
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.fps_cache = {}

        self.face_base_dir = face_base_dir
        self.video_base_dir = video_base_dir
        self.audio_base_dir = audio_base_dir

        self.face_path = face_path
        self.labels_path = labels_path
        self.detect_path = detect_path

        self.talker_face_map = None
        self.load_talker_face_map()

        if type(file_map) in (str, list):
            file_map = self.load_folders(file_map)

        self.file_map = file_map

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
        for k in tqdm(range(len(file_column))):
            filename = file_column[k]
            is_fake = labels_map[filename]
            is_talker = talker_column[k]
            num_faces = num_faces_column[k]
            face_no = face_column[k]

            if is_fake:
                continue
            elif not is_talker and (num_faces > 1):
                continue

            talker_face_map[filename] = face_no

        self.talker_face_map = talker_face_map
        return talker_face_map

    def load_folders(self, folders):
        if type(folders) is str:
            basedir = folders
            folders = os.listdir(folders)
        else:
            basedir = None

        file_map = {}

        for folder in folders:
            folder_path = f'{basedir}/{folder}'
            filenames = os.listdir(folder_path)
            video_filename = f'{folder}.mp4'
            if video_filename not in self.talker_face_map:
                continue

            talker_face_no = self.talker_face_map[video_filename]
            allowed_image_paths = []

            for filename in filenames:
                face_no, frame_no = self.extract_frame(filename)
                if talker_face_no != face_no:
                    continue

                if basedir is not None:
                    image_path = f'{basedir}/{folder}/{filename}'
                else:
                    image_path = f'{folder}/{filename}'

                allowed_image_paths.append(image_path)

            file_map[folder] = allowed_image_paths

        return file_map

    def choose_random_filename(self):
        filenames = list(self.file_map.keys())
        filename = random.choice(filenames)
        return filename

    def load_image_paths(self, filename, randomize_images=False):
        image_paths = self.file_map[filename]
        if randomize_images:
            image_paths = random.sample(
                image_paths, k=len(image_paths)
            )

        return image_paths

    def _load_random_video(self, randomize_images=True):
        filename = self.choose_random_filename()
        image_paths = self.load_image_paths(
            filename, randomize_images=randomize_images
        )

        orig_mel = self.load_audio(filename)
        fps = self.resolve_fps(filename)
        return orig_mel, image_paths, fps

    def load_audio(self, filename):
        filename = os.path.basename(filename)
        name = filename

        try:
            name = name[:name.rindex('.')]
        except ValueError:
            pass

        audio_path = f'{self.audio_base_dir}/{name}.flac'
        wav = audio.load_wav(audio_path, hparams.sample_rate)
        orig_mel = audio.melspectrogram(wav).T
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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