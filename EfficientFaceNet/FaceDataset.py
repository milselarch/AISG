try:
    import ParentImport

    from FaceAnalysis import FaceCluster

except ModuleNotFoundError:
    from . import ParentImport

    from ..FaceAnalysis import FaceCluster

import random
import numpy as np
import pandas as pd
import os

from tqdm.auto import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from datetime import datetime as Datetime
from PIL import Image, ImageOps

class FaceDataset(object):
    def __init__(self, train_size=0.9, seed=42, load=True):
        self.train_size = train_size
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )]
        )

        self.face_path = '../stats/all-labels.csv'
        # self.detect_path = '../stats/sorted-detections.csv'
        self.detect_path = '../stats/mtcnn/labelled-mtcnn.csv'
        self.face_dir = '../datasets/extract/mtcnn-sync'

        self.real_files = None
        self.fake_files = None
        self.img_labels = None

        self.all_filenames = None
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

        self.real_train_files = None
        self.fake_train_files = None
        self.real_test_files = None
        self.fake_test_files = None

        self.talker_face_map = None

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

        # face_path = '../stats/face-predictions-211022-1416.csv'
        face_path = '../stats/face-predictions-211106-0109.csv'
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

    def make_image_map(self, image_names):
        face_map = {}

        for image_name in image_names:
            face_no, frame_no = self.extract_frame(image_name)

            if face_no not in face_map:
                face_map[face_no] = []

            face_map[face_no].append(image_name)

        return face_map

    def label_all_frames(
        self, predict, tag=None, max_samples=None,
        clip=None
    ):
        if tag is None:
            tag = self.make_date_stamp()

        # face_cluster = FaceCluster(load_datasets=False)
        # face_path = self.face_path
        # face_map = face_cluster.make_face_map(face_path)
        detect_path = self.detect_path
        detections = pd.read_csv(detect_path)

        filenames = detections['filename']
        filenames = np.unique(filenames.to_numpy())
        filename_log, prediction_log = [], []
        face_log, frame_log = [], []

        if clip is not None:
            filenames = filenames[:clip]

        for filename in tqdm(filenames):
            name = filename[:filename.rindex('.')]
            image_dir = f'{self.face_dir}/{name}'
            image_names = os.listdir(image_dir)
            face_image_map = self.make_image_map(image_names)

            for face_no in face_image_map:
                face_image_names = face_image_map[face_no]

                if max_samples is not None:
                    random.shuffle(face_image_names)
                    face_image_names = face_image_names[:max_samples]

                for image_name in face_image_names:
                    frame_no = self.get_frame_no(image_name)
                    img_file = f'{name}/{face_no}-{frame_no}.jpg'
                    img_path = f'{self.face_dir}/{img_file}'

                    prediction = predict(img_path)
                    prediction_log.append(prediction)
                    filename_log.append(filename)
                    frame_log.append(frame_no)
                    face_log.append(face_no)

        df = pd.DataFrame(data={
            'filename': filename_log, 'face': face_log,
            'frame': frame_log, 'pred': prediction_log
        })

        path = f'stats/face-predictions-{tag}.csv'
        df.to_csv(path, index=False)
        print(f'face predictions exported to {path}')

    def load_talker_face_map(self):
        # face_cluster = FaceCluster(load_datasets=False)
        # labels_map = face_cluster.get_orig_labels(self.labels_path)

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
            # is_fake = labels_map[filename]
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

    @staticmethod
    def extract_frame(filename):
        base_filename = os.path.basename(filename)
        name = base_filename[:base_filename.index('.')]
        face_no, frame_no = [int(x) for x in name.split('-')]
        return face_no, frame_no

    @classmethod
    def get_frame_no(cls, filename):
        face_no, frame_no = cls.extract_frame(filename)
        return frame_no

    @classmethod
    def get_face_no(cls, filename):
        face_no, frame_no = cls.extract_frame(filename)
        return face_no

    def load_datasets(self):
        if self.talker_face_map is None:
            self.load_talker_face_map()

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

        image_paths_map = {}
        skipped_files, trainable_files = 0, 0
        self.all_filenames = []
        # self.real_filenames = []
        # self.fake_filenames = []

        for item in os.walk(self.face_dir):
            folder_path, dirs, image_names = item
            # folder_name = os.path.basename(folder_path)
            image_paths_map[folder_path] = image_names

        for filename in tqdm(face_map):
            if filename not in file_map:
                continue

            face_fake = face_map[filename]
            frames = file_map[filename]
            unique_face_nos = frames[0]['num_faces']

            if face_fake in (0, 1):
                self.all_filenames.append(filename)
                trainable_files += 1
            else:
                skipped_files += 1
                continue

            if unique_face_nos == 2:
                self.real_files[filename] = []
                self.fake_files[filename] = []
            elif face_fake == 0:
                self.real_files[filename] = []
            elif face_fake == 1:
                self.fake_files[filename] = []
            else:
                continue

            name = filename[:filename.rindex('.')]
            face_folder_path = f'{self.face_dir}/{name}'
            image_names = image_paths_map[face_folder_path]
            # image_names = os.listdir(face_folder_path)

            image_face_nos = np.unique([
                self.get_face_no(image_name)
                for image_name in image_names
            ])

            assert unique_face_nos < 3
            talker_face_no = face_map[filename]
            num_image_face_nos = len(image_face_nos)

            if num_image_face_nos == 2:
                assert unique_face_nos == num_image_face_nos

            for k, image_name in enumerate(image_names):
                img_path = f'{face_folder_path}/{image_name}'
                face_no, frame_no = self.extract_frame(image_name)
                is_talker = face_no == talker_face_no
                # img_file = f'{name}/{face_no}-{frame_no}.jpg'
                # img_path = f'{self.face_dir}/{img_file}'

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

        print(f'trainable files: {trainable_files}')
        print(f'skipped files: {skipped_files}')
        self.train_test_split()

    def train_test_split(self):
        self.real_train_files = []
        self.fake_train_files = []
        self.real_test_files = []
        self.fake_test_files = []

        self.real_train_images = []
        self.fake_train_images = []
        self.real_test_images = []
        self.fake_test_images = []

        fake_labels = [1] * len(self.all_filenames)
        x_train, x_test, _, _ = train_test_split(
            self.all_filenames, fake_labels,
            random_state=self.seed, train_size=self.train_size
        )

        self.train_files = x_train
        self.test_files = x_test

        for filename in self.all_filenames:
            fake_image_paths, real_image_paths = [], []

            if filename in self.fake_files:
                fake_image_paths = self.fake_files[filename]
            if filename in self.real_files:
                real_image_paths = self.real_files[filename]

            if filename in self.train_files:
                self.fake_train_images.extend(fake_image_paths)
                self.real_train_images.extend(real_image_paths)

                if len(real_image_paths) > 0:
                    self.real_train_files.append(filename)
                if len(fake_image_paths) > 0:
                    self.fake_train_files.append(filename)

            elif filename in self.test_files:
                self.fake_test_images.extend(fake_image_paths)
                self.real_test_images.extend(real_image_paths)

                if len(real_image_paths) > 0:
                    self.real_test_files.append(filename)
                if len(fake_image_paths) > 0:
                    self.fake_test_files.append(filename)

            else:
                raise ValueError(
                    f'{filename} NOT IN TRAIN / TEST'
                )

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

        real_paths = self.load_image_paths(
            0, samples=real_count, is_training=is_training
        )
        fake_paths = self.load_image_paths(
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

    def load_image_path(self, label, is_training=True):
        assert label in (0, 1)

        if label == 0:
            file_map = self.real_files

            if is_training:
                filenames = self.real_train_files
            else:
                filenames = self.real_test_files
        else:
            file_map = self.fake_files

            if is_training:
                filenames = self.fake_train_files
            else:
                filenames = self.fake_test_files

        filename = random.choice(filenames)
        image_paths = file_map[filename]
        image_path = random.choice(image_paths)
        return image_path

    def load_image_paths(self, *args, **kwargs):
        return self.load_image_paths_v2(*args, **kwargs)

    def load_image_paths_v2(
        self, label, samples=1, is_training=True
    ):
        assert label in (0, 1)
        img_samples = []

        for k in range(samples):
            img_samples.append(self.load_image_path(
                label, is_training=is_training
            ))

        return img_samples

    def load_image_paths_v1(
        self, label, samples=1, is_training=True
    ):
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

        img_samples = random.sample(filenames, samples)
        return img_samples