import ParentImport

import random
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torchvision import transforms
from FaceAnalysis import FaceCluster
from sklearn.model_selection import train_test_split
from PIL import Image

class Dataset(object):
    def __init__(self, train_size=0.9, seed=42, load=True):
        self.train_size = train_size
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.real_files = None
        self.fake_files = None
        self.img_labels = None
        self.real_filenames = None
        self.fake_filenames = None

        self.train_files = None
        self.train_labels = None
        self.test_files = None
        self.test_labels = None

        self.real_train_images = None
        self.fake_train_images = None
        self.real_test_images = None
        self.fake_test_images = None

        if load:
            self.load_datasets()

    def load_datasets(self):
        face_cluster = FaceCluster(load_datasets=False)
        face_path = '../stats/bg-clusters/face-vid-labels.csv'
        face_map = face_cluster.make_face_map(face_path)
        detect_path = '../stats/sorted-detections.csv'
        detections = pd.read_csv(detect_path)

        self.real_files = {}
        self.fake_files = {}
        self.img_labels = {}

        file_column = detections['filename'].to_numpy()
        face_column = detections['face'].to_numpy()
        frame_column = detections['frame'].to_numpy()
        talker_column = detections['talker'].to_numpy()
        num_faces_column = detections['num_faces'].to_numpy()
        file_map = {}

        for k in tqdm(range(len(file_column))):
            filename = file_column[k]
            if filename not in file_map:
                file_map[filename] = []

            frames = file_map[filename]
            frames.append({
                'face': face_column[k],
                'frame': frame_column[k],
                'talker': talker_column[k],
                'num_faces': num_faces_column[k]
            })

        for filename in tqdm(face_map):
            if filename not in file_map:
                continue

            face_fake = face_map[filename]
            frames = file_map[filename]
            unique_face_nos = frames[0]['num_faces']

            if unique_face_nos == 2:
                self.real_files[filename] = []
                self.fake_files[filename] = []
            elif face_fake == 0:
                self.real_files[filename] = []
            elif face_fake == 1:
                self.fake_files[filename] = []
            else:
                continue

            for k in range(len(frames)):
                frame_no = frames[k]['frame']
                is_talker = frames[k]['talker']
                face_no = frames[k]['face']

                name = filename[:filename.index('.')]
                img_file = f'{name}/{face_no}-{frame_no}.jpg'
                img_path = f'../datasets-local/faces/{img_file}'

                if face_fake == 0:
                    label = 0
                elif unique_face_nos == 2:
                    label = 1 if is_talker else 0
                else:
                    label = 1

                if label == 1:
                    self.fake_files[filename].append(img_path)
                else:
                    self.real_files[filename].append(img_path)

                self.img_labels[img_path] = label

        self.train_test_split()

    def train_test_split(self):
        all_filenames = list(self.img_labels.keys())
        all_labels = list(self.img_labels.values())
        x_train, x_test, y_train, y_test = train_test_split(
            all_filenames, all_labels,
            random_state=self.seed, train_size=self.train_size
        )

        self.train_files = x_train
        self.train_labels = y_train
        self.test_files = x_test
        self.test_labels = y_test

        self.real_train_images = []
        self.fake_train_images = []
        self.real_test_images = []
        self.fake_test_images = []

        for img_path, label in tqdm(zip(x_train, y_train)):
            if label == 0:
                self.real_train_images.append(img_path)
            elif label == 1:
                self.fake_train_images.append(img_path)

        for img_path, label in tqdm(zip(x_test, y_test)):
            if label == 0:
                self.real_test_images.append(img_path)
            elif label == 1:
                self.fake_test_images.append(img_path)

        print(f'real train images {len(self.real_train_images)}')
        print(f'fake train images {len(self.fake_train_images)}')
        print(f'real test images {len(self.real_test_images)}')
        print(f'fake test images {len(self.fake_test_images)}')
        print(f'loaded')

    def prepare_batch(
        self, batch_size=32, fake_p=0.5, is_training=True,
        randomize=False
    ):
        fake_count = int(batch_size * fake_p)
        real_count = batch_size - fake_count

        real_paths = self.load_filenames(
            0, samples=real_count, is_training=is_training
        )
        fake_paths = self.load_filenames(
            1, samples=fake_count, is_training=is_training
        )

        labels = [0] * real_count + [1] * fake_count
        all_paths = real_paths + fake_paths
        images = []

        for file_path in all_paths:
            image = self.pil_loader(file_path)
            image = self.transform(image)
            # images.append(image)
            expanded_image = np.expand_dims(image, axis=0)
            images.append(expanded_image)

        np_images = np.vstack(images)
        np_labels = np.array(labels)

        if randomize:
            indexes = np.arange(len(np_labels))
            np.random.shuffle(indexes)
            np_labels = np_labels[indexes]
            np_images = np_images[indexes]

        np_labels = np.expand_dims(np_labels, axis=1)
        return np_images, np_labels

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def load_filenames(self, label, samples=1, is_training=True):
        assert label in (0, 1)

        if is_training:
            if label == 0:
                filenames = self.real_train_images
            else:
                filenames = self.fake_train_images
        else:
            if label == 0:
                filenames = self.real_test_images
            else:
                filenames = self.fake_test_images

        samples = random.sample(filenames, samples)
        return samples