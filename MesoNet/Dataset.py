try:
    import ParentImport

    from FaceAnalysis import FaceCluster

except ModuleNotFoundError:
    from . import ParentImport

    from ..FaceAnalysis import FaceCluster

import random
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from datetime import datetime as Datetime
from PIL import Image, ImageOps

class Dataset(object):
    def __init__(self, train_size=0.9, seed=42, load=True):
        self.train_size = train_size
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.face_path = '../stats/all-labels.csv'
        # self.detect_path = '../stats/sorted-detections.csv'
        self.detect_path = '../stats/labelled-mtcnn.csv'
        self.face_dir = '../datasets-local/mtcnn-faces'

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

    @staticmethod
    def make_date_stamp():
        return Datetime.now().strftime("%y%m%d-%H%M")

    @staticmethod
    def collate_preds(pred_rows, roll):
        preds = pred_rows['prediction'].to_numpy()

        median_pred = np.median(preds)
        quartile_pred_3 = np.percentile(sorted(preds), 75)
        quartile_pred_1 = np.percentile(sorted(preds), 25)

        roll_pred = pd.Series(preds).rolling(roll).median()
        roll_pred = roll_pred.to_numpy()
        roll_pred = roll_pred[~np.isnan(roll_pred)]

        try:
            group_pred = np.percentile(sorted(roll_pred), 75)
        except IndexError:
            group_pred = quartile_pred_3

        return np.array([
            median_pred, quartile_pred_1, quartile_pred_3, group_pred
        ])

    def label_all_videos(self, tag=None, roll=3):
        if tag is None:
            tag = self.make_date_stamp()

        face_path = '../stats/face-predictions-211022-1416.csv'
        labels_path = '../stats/all-labels.csv'
        face_df = pd.read_csv(face_path)
        labels_df = pd.read_csv(labels_path)

        for index in tqdm(labels_df.index):
            row = labels_df.loc[index]
            filename = row['filename']
            pred_rows = face_df[face_df['filename'] == filename]

            if len(pred_rows) == 0:
                labels_df.loc[index, 'median'] = 0.85
                labels_df.loc[index, '1st_quartile_pred'] = 0.8
                labels_df.loc[index, '3rd_quartile_pred'] = 0.9
                labels_df.loc[index, 'group_pred'] = 0.9
                continue

            face_nos = pred_rows['face'].to_numpy()
            face_nos = np.unique(face_nos)
            face_preds = []

            for face_no in face_nos:
                face_rows = pred_rows[pred_rows['face'] == face_no]
                preds = self.collate_preds(face_rows, roll)
                face_preds.append(preds)

            face_preds = np.array(face_preds)
            face_preds = np.max(face_preds, axis=0)

            labels_df.loc[index, 'median'] = face_preds[0]
            labels_df.loc[index, '1st_quartile_pred'] = face_preds[1]
            labels_df.loc[index, '3rd_quartile_pred'] = face_preds[2]
            labels_df.loc[index, 'group_pred'] = face_preds[3]

        path = f'../stats/vid-face-preds-{tag}.csv'
        labels_df.to_csv(path, index=False)
        print(f'video face predictions exported to {path}')

    def label_all_frames(self, predict, tag=None):
        if tag is None:
            tag = self.make_date_stamp()

        face_cluster = FaceCluster(load_datasets=False)
        face_path = self.face_path
        face_map = face_cluster.make_face_map(face_path)
        detect_path = self.detect_path
        detections = pd.read_csv(detect_path)

        for index in tqdm(detections.index):
            row = detections.loc[index]
            filename = row['filename']
            frame_no = row['frame']
            face_no = row['face']

            name = filename[:filename.index('.')]
            img_file = f'{name}/{face_no}-{frame_no}.jpg'
            img_path = f'{self.face_dir}/{img_file}'
            prediction = predict(img_path)
            detections.loc[index, 'prediction'] = prediction

        path = f'../stats/face-predictions-{tag}.csv'
        detections.to_csv(path, index=False)
        print(f'face predictions exported to {path}')

    def load_datasets(self):
        face_cluster = FaceCluster(load_datasets=False)
        face_path = self.face_path
        face_map = face_cluster.make_face_map(face_path)
        detect_path = self.detect_path
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
                img_path = f'{self.face_dir}/{img_file}'
                assert unique_face_nos < 3

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
        np_images, np_labels = self.load_batch(
            all_paths, labels, randomize=randomize
        )

        return np_images, np_labels

    def load_batch(
        self, batch_filepaths, batch_labels, randomize=False
    ):
        images = []

        for file_path in batch_filepaths:
            image = self.pil_loader(file_path)
            image = self.transform(image)
            # images.append(image)
            expanded_image = np.expand_dims(image, axis=0)
            images.append(expanded_image)

        np_images = np.vstack(images)
        np_labels = np.array(batch_labels)

        if randomize:
            indexes = np.arange(len(np_labels))
            np.random.shuffle(indexes)
            np_labels = np_labels[indexes]
            np_images = np_images[indexes]

        np_labels = np.expand_dims(np_labels, axis=1)
        return np_images, np_labels

    @staticmethod
    def pil_loader(path: str, mirror_prob=0.5) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            if random.random() < mirror_prob:
                img = ImageOps.mirror(img)

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