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

from tqdm import tqdm
from torch import nn
from torch import optim
from glob import glob
from hparams import hparams, get_image_list
from torch.utils import data as data_utils
from models import SyncNet_color as SyncNet
from os.path import dirname, join, basename, isfile

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(
        self, seed=32, train_size=0.95, load=True,
        mel_step_size=16
    ):
        self.mel_step_size = mel_step_size

        self.face_base_dir = '../datasets-local/faces'
        self.video_base_dir = '../datasets/train/videos'

        self.face_files = None
        self.train_face_files = None
        self.test_face_files = None

        self.train_size = train_size
        self.seed = seed

        self.real_sample_cache = {}
        self.fake_sample_cache = {}
        self.fps_cache = {}

        if load:
            self.load_datasets()

    def load_datasets(self):
        face_cluster = FaceCluster(load_datasets=False)
        face_path = '../stats/bg-clusters/face-vid-labels.csv'
        face_map = face_cluster.make_face_map(face_path)
        labels_map = face_cluster.get_orig_labels()
        detect_path = '../stats/sorted-detections.csv'
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

    def load_real_samples(
        self, filename, is_training=True
    ):
        if is_training:
            image_paths = self.train_face_files[filename]
        else:
            image_paths = self.test_face_files[filename]

        if filename not in self.fps_cache:
            filepath = f'{self.video_base_dir}/{filename}'
            cap = cv2.VideoCapture(filepath)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.fps_cache[filename] = fps

        fps = self.fps_cache[filename]
        for image_path in image_paths:
            image = cv2.imread(image_path)



    @staticmethod
    def get_frame_no(filename):
        base_filename = basename(filename)
        name = base_filename[:base_filename.index('.')]
        face_no, frame_no = [int(x) for x in name.split('-')]
        return face_no, frame_no

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vid_name = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vid_name, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None

            window_fnames.append(frame)

        return window_fnames

    def crop_audio_window(self, spec, start_frame, fps):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(fps)))

        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx: end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(
                        img, (hparams.img_size, hparams.img_size)
                    )
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if mel.shape[0] != syncnet_mel_step_size:
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y