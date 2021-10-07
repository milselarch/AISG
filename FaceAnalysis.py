import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datasets
import imagehash
import cv2
import os

from datetime import datetime
from tqdm.auto import tqdm
from PIL import Image


class FaceCluster(object):
    def __init__(
        self, labels_path='../datasets/extra-labels.csv',
        durations_path='stats/aisg-durations-210929-0931.csv',
    ):
        self.cache = {}
        self.labels_path = labels_path
        self.durations_path = durations_path
        self.stamp = self.make_date_stamp()

        self.video_basedir = 'datasets/train/videos'
        self.base_filename = "detections-20210903-230613.csv"
        self.base_faces = pd.read_csv(f'stats/{self.base_filename}')
        self.dataset = datasets.Dataset()
        self.duration_map = None

    @staticmethod
    def make_date_stamp():
        return datetime.now().strftime("%y%m%d-%H%M")

    def build_duration_map(self):
        df = pd.read_csv(self.durations_path)
        filenames = df['filename'].to_numpy().tolist()
        labels = df['label'].to_numpy()
        durations = df['duration'].to_numpy()

        duration_map = {}
        pbar = tqdm(range(len(filenames)))

        for k in pbar:
            filename = filenames[k]
            duration = durations[k]
            label = labels[k]

            name = filename[:filename.index('.')]
            file_path = f'{self.video_basedir}/{name}.mp4'

            if not os.path.exists(file_path):
                continue

            # print('DURATION', duration)
            # df.loc[cond, 'duration'] = duration
            desc = f'{filename} - {duration}'
            pbar.set_description(desc)

            if duration not in duration_map:
                duration_map[duration] = {'real': [], 'fake': []}

            str_label = 'fake' if label == 1 else 'real'
            duration_map[duration][str_label].append(filename)

        self.duration_map = duration_map
        return duration_map

    def resolve(
        self, filename, frame_no, cache=True,
        scale=1
    ):
        try:
            name = filename[:filename.index('.')]
        except ValueError as e:
            print('bad filename', filename)
            raise e

        file_path = f'{self.video_basedir}/{name}.mp4'

        if not os.path.exists(file_path):
            invalid.append(file_path)
            raise FileNotFoundError

        key = (name, frame_no)
        if key in self.cache:
            return self.cache[name]

        video_obj = self.dataset.read_video(
            filename, specific_frames=[frame_no],
            scale=scale
        )

        frame = video_obj.out_video[0]

        if cache:
            self.cache[name] = frame

        return frame

    def get_start_frame_no(self, filename):
        video_frame_rows = self.base_faces[
            self.base_faces['filename'] == filename
        ]

        num_faces = video_frame_rows['num_faces'].to_numpy()

        try:
            num_faces = num_faces[0]
        except IndexError:
            return None

        frames = video_frame_rows['frames'].to_numpy()
        frames = np.unique(frames)

        for frame_no in frames:
            face_frames = video_frame_rows[
                video_frame_rows['frames'] == frame_no
            ]

            if len(face_frames) == num_faces:
                return frame_no

    def get_face_areas(self, filename, frame_no, scale=1):
        face_frames = self.base_faces[
            (self.base_faces['filename'] == filename) &
            (self.base_faces['frames'] == frame_no)
        ]

        face_areas = []
        for index in face_frames.index:
            frame_data = face_frames.loc[index]

            true_top = int(frame_data['top'])
            true_left = int(frame_data['left'])
            true_right = int(frame_data['right'])
            true_bottom = int(frame_data['bottom'])
            area = (true_bottom - true_top) * (true_right - true_left)
            buffer = area ** 0.5

            top = max(int((true_top - buffer) * scale), 0)
            left = max(int((true_left - buffer) * scale), 0)
            right = max(int((true_right + buffer) * scale), 0)
            bottom = max(int((true_bottom + buffer) * scale), 0)

            coords = (top, left, right, bottom)
            face_areas.append(coords)

        face_areas = tuple(face_areas)
        return face_areas

    @staticmethod
    def threshold(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 11
        )

    @classmethod
    def make_hashes(cls, image, face_areas):
        total_face_hash = ''

        for face_area in face_areas:
            top, left, right, bottom = face_area
            face_crop = image[top:bottom, left:right]

            pil_face_img = Image.fromarray(face_crop)
            face_crop_hash = imagehash.phash(pil_face_img)
            total_face_hash += str(face_crop_hash)

            image[top:bottom, left:right, :] = 0

        # image = cls.threshold(image)
        pil_bg_image = Image.fromarray(image)
        background_hash = imagehash.phash(pil_bg_image)
        return background_hash, total_face_hash

    def clear_cache(self):
        self.cache = {}

    def make_background_clusters(
        self, files, pbar, scale=1, threshold=6
    ):
        self.clear_cache()
        clusters = []
        distances = []
        background_hashes = []
        face_hashes = []

        while len(files) > 0:
            base_filename = files[0]
            frame_no = self.get_start_frame_no(base_filename)
            new_files = []

            if frame_no is None:
                clusters.append([base_filename])
                background_hashes.append([0])
                distances.append([0])
                face_hashes.append([0])
                continue

            base_image = self.resolve(
                base_filename, frame_no, scale=scale
            )
            face_areas = self.get_face_areas(
                base_filename, frame_no, scale=scale
            )
            base_bg_hash, base_face_hash = self.make_hashes(
                base_image, face_areas
            )

            face_hash_batch = [base_face_hash]
            background_hash_batch = [base_bg_hash]
            cluster = [base_filename]
            distance_batch = [0]
            pbar.update()

            for filename in files[1:]:
                image = self.resolve(filename, frame_no)

                if image.shape != base_image.shape:
                    new_files.append(filename)
                    continue

                bg_hash, face_hash = self.make_hashes(
                    image, face_areas
                )
                distance = base_bg_hash - bg_hash

                if distance > threshold:
                    new_files.append(filename)
                else:
                    cluster.append(filename)
                    background_hash_batch.append(bg_hash)
                    face_hash_batch.append(face_hash)
                    distance_batch.append(distance)
                    pbar.update()

            clusters.append(cluster)
            distances.append(distance_batch)
            background_hashes.append(background_hash_batch)
            face_hashes.append(face_hash_batch)
            # print(base_filename, files[1:])
            # print(cluster_dist)
            files = new_files

        # print(clusters, distances)
        return clusters, distances, background_hashes, face_hashes

    def cluster(self, max_durations=None, threshold=6):
        duration_map = self.build_duration_map()
        num_files = sum([
            len(duration_map[duration]['real']) +
            len(duration_map[duration]['fake'])
            for duration in duration_map
        ])

        cluster_no = 0
        cluster_log, label_log = [], []
        bg_hash_log, filename_log = [], []
        face_hash_log, distance_log = [], []

        pbar = tqdm(range(num_files))
        durations = list(duration_map.keys())
        if max_durations is not None:
            durations = durations[:max_durations]

        for duration in durations:
            mapping = duration_map[duration]
            reals = mapping['real']
            fakes = mapping['fake']
            files = reals + fakes

            result = self.make_background_clusters(
                files, pbar, threshold=threshold
            )

            clusters, distances, bg_hashes, face_hashes = result
            # print('CLUSTERS', clusters)
            # print('DISTANCES', distances)

            for k, cluster in enumerate(clusters):
                bg_hash_batch = bg_hashes[k]
                face_hash_batch = face_hashes[k]
                distance_batch = distances[k]
                assert len(bg_hash_batch) == len(cluster)

                for i, filename in enumerate(cluster):
                    label = 0 if filename in reals else 1
                    face_hash = face_hash_batch[i]
                    distance = distance_batch[i]
                    bg_hash = bg_hash_batch[i]

                    filename_log.append(filename)
                    bg_hash_log.append(bg_hash)
                    distance_log.append(distance)
                    face_hash_log.append(face_hash)
                    cluster_log.append(cluster_no)
                    label_log.append(label)

                cluster_no += 1
                desc = f'clusters: {cluster_no}'
                pbar.set_description(desc)

        cluster_df = pd.DataFrame(data={
            'cluster': cluster_log, 'filename': filename_log,
            'distance': distance_log, 'bg_hash': bg_hash_log,
            'label': label_log, 'face_hash': face_hash_log
        })

        basedir = f'stats/bg-clusters'
        path = f'{basedir}/cluster-{self.stamp}.csv'
        cluster_df.to_csv(path, index=False)
        print(f'clusters saved at {path}')
        return cluster_df