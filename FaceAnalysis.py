import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datasets
import imagehash
import loader
import cv2
import os

from datetime import datetime
from tqdm.auto import tqdm
from PIL import Image


class FaceCluster(object):
    def __init__(
        self, labels_path='../datasets/extra-labels.csv',
        durations_path='stats/aisg-durations-210929-0931.csv',
        # cluster_path='stats/bg-clusters/cluster-211008-0015.csv'
        cluster_path='stats/bg-clusters/cluster-211008-1233.csv'
    ):
        self.cache = {}
        self.labels_path = labels_path
        self.durations_path = durations_path
        self.stamp = self.make_date_stamp()

        self.cluster_path = cluster_path
        # self.cluster_df = pd.read_csv(cluster_path)

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

        bounding_box = video_obj.cut_blackout()
        frame = video_obj.out_video[0]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = (gray_frame, bounding_box)

        if cache:
            self.cache[name] = result

        return result

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

            if len(image.shape) == 3:
                image[top:bottom, left:right, :] = 0
            else:
                assert len(image.shape) == 2
                image[top:bottom, left:right] = 0

        # image = cls.threshold(image)
        pil_bg_image = Image.fromarray(image)
        background_hash = imagehash.phash(pil_bg_image)
        return background_hash, total_face_hash

    def clear_cache(self):
        self.cache = {}

    def make_background_clusters(
        self, files, pbar=None, scale=1, threshold=6
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
                files = files[1:]
                continue

            result = self.extract(base_filename, frame_no, scale)
            base_image, face_areas = result
            base_bg_hash, base_face_hash = self.make_hashes(
                base_image, face_areas
            )

            face_hash_batch = [base_face_hash]
            background_hash_batch = [base_bg_hash]
            cluster = [base_filename]
            distance_batch = [0]

            if pbar is not None:
                pbar.update()

            for filename in files[1:]:
                image, image_bounds = self.resolve(
                    filename, frame_no
                )

                if image.shape != base_image.shape:
                    new_files.append(filename)
                    continue

                image = self.resize_frame(image, image_bounds)
                hashes = self.make_hashes(image, face_areas)
                distance = base_bg_hash - bg_hash

                if distance > threshold:
                    new_files.append(filename)
                else:
                    cluster.append(filename)
                    background_hash_batch.append(bg_hash)
                    face_hash_batch.append(face_hash)
                    distance_batch.append(distance)

                    if pbar is not None:
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

    @staticmethod
    def resize_face_areas(image, bounds, face_areas):
        new_face_areas = []
        height = image.shape[0]
        width = image.shape[1]

        for face_area in face_areas:
            top, left, right, bottom = face_area
            new_face_areas.append((
                bounds.rescale_y(top, height),
                bounds.rescale_x(left, width),
                bounds.rescale_x(right, width),
                bounds.rescale_y(bottom, height)
            ))

        return tuple(new_face_areas)

    @staticmethod
    def resize_frame(frame, bounding_box):
        x_start, x_end, y_start, y_end = bounding_box.to_tuple()
        cropped_frame = frame[y_start:y_end, x_start:x_end]

        pil_image = Image.fromarray(cropped_frame)
        resized_frame = pil_image.resize(
            frame.shape[::-1], Image.BICUBIC
        )

        resized_frame = np.array(resized_frame)
        assert resized_frame.shape == frame.shape
        return resized_frame

    def extract(self, base_filename, frame_no, scale=1):
        raw_image, base_bounds = self.resolve(
            base_filename, frame_no, scale=scale
        )
        raw_face_areas = self.get_face_areas(
            base_filename, frame_no, scale=scale
        )

        base_image = self.resize_frame(raw_image, base_bounds)
        face_areas = self.resize_face_areas(
            base_image, base_bounds, raw_face_areas
        )

        return base_image, face_areas

    def get_clusters_info(self, verbose=True):
        cluster_df = pd.read_csv(self.cluster_path)
        cluster_nos = cluster_df['cluster'].to_numpy()
        cluster_nos = np.unique(cluster_nos)

        real_clusters, fake_clusters = {}, {}
        mixed_clusters = {}

        for cluster_no in cluster_nos:
            cluster_rows = cluster_df[
                cluster_df['cluster'] == cluster_no
            ]

            labels = cluster_rows['label'].to_numpy()
            real_only = max(labels) == 0
            fake_only = min(labels) == 1

            if real_only:
                real_clusters[cluster_no] = cluster_rows
            elif fake_only:
                fake_clusters[cluster_no] = cluster_rows
            else:
                mixed_clusters[cluster_no] = cluster_rows

        if verbose:
            print(cluster_df)
            print('REAL CLUSTERS', len(real_clusters))
            print('FAKE CLUSTERS', len(fake_clusters))
            print('MIXED CLUSTERS', len(mixed_clusters))
            print('END')

        return real_clusters, fake_clusters, mixed_clusters

    def cross_clusters(self, num_clusters=None, scale=1):
        cluster_groups = self.get_clusters_info(verbose=False)
        real_clusters, fake_clusters, mixed_clusters = cluster_groups
        sub_real_clusters = {**real_clusters, **mixed_clusters}
        cluster_log, nearest_distance_log = [], []
        nearest_file_log, nearest_cluster_log = [], []
        filename_log = []

        fake_cluster_nos = list(fake_clusters.keys())
        if num_clusters is not None:
            fake_cluster_nos = fake_cluster_nos[:num_clusters]

        pbar = tqdm(fake_cluster_nos)

        for fake_cluster_no in pbar:
            fake_cluster = fake_clusters[fake_cluster_no]
            row = fake_cluster.loc[fake_cluster.index[0]]
            base_filename = row['filename']
            frame_no = self.get_start_frame_no(base_filename)
            # cluster_log.append(fake_cluster_no)

            if frame_no is None:
                cluster_log.append(fake_cluster_no)
                filename_log.append(base_filename)
                nearest_distance_log.append(-1)
                nearest_cluster_log.append([])
                nearest_file_log.append([])
                continue

            result = self.extract(base_filename, frame_no, scale)
            base_image, base_face_areas = result
            base_bg_hash, base_face_hash = self.make_hashes(
                base_image, base_face_areas
            )

            nearest_clusters = []
            nearest_filenames = []
            closest_distance = float('inf')

            for i, real_cluster_no in enumerate(sub_real_clusters):
                real_cluster = sub_real_clusters[real_cluster_no]
                str_hash = real_cluster['bg_hash'].to_numpy()[0]
                assert type(str_hash) == str
                """
                if str_hash == '0':
                    # video with no faces detected
                    continue
                """
                row = None
                for index in real_cluster.index:
                    row = real_cluster.loc[index]
                    if row['label'] == 0:
                        break

                assert row is not None
                assert row['label'] == 0
                filename = row['filename']

                try:
                    result = self.extract(filename, frame_no, scale)
                except datasets.FailedVideoRead:
                    continue

                image, face_areas = result
                if base_image.shape != image.shape:
                    continue
                """
                if face_areas is None:
                    continue
                """
                hashes = self.make_hashes(image, base_face_areas)
                bg_hash, face_hash = hashes
                distance = bg_hash - base_bg_hash

                if distance < closest_distance:
                    nearest_clusters = [real_cluster_no]
                    nearest_filenames = [filename]
                    closest_distance = distance
                elif distance == closest_distance:
                    nearest_clusters.append(real_cluster_no)
                    nearest_filenames.append(filename)

                length = len(sub_real_clusters)
                file_pair = f'{base_filename} {filename}'
                distance_pair = f'[{distance}][{closest_distance}]'
                sub_progress = f'[{i}/{length}]'

                desc = f'{file_pair} {sub_progress} {distance_pair}'
                pbar.set_description(desc)

            filename_log.append(base_filename)
            cluster_log.append(fake_cluster_no)
            nearest_distance_log.append(closest_distance)
            nearest_cluster_log.append(nearest_clusters)
            nearest_file_log.append(nearest_filenames)

        basedir = f'stats/bg-clusters'
        path = f'{basedir}/cross-clusters-{self.stamp}.csv'
        cluster_df = pd.DataFrame(data={
            'cluster': cluster_log, 'filename': filename_log,
            'nearest_distance': nearest_distance_log,
            'nearest_clusters': nearest_cluster_log,
            'nearest_files': nearest_file_log
        })

        cluster_df.to_csv(path, index=False)
        print(f'clusters saved at {path}')
        return cluster_df

    @staticmethod
    def thumbnail_videos(cluster_no, filenames):
        plt.cla()
        pbar = tqdm(range(len(filenames)))

        plt.title(cluster_no)
        plt.show()

        for k in pbar:
            filename = filenames[k]
            pbar.set_description(filename)
            path = f'datasets/train/videos/{filename}'
            out = loader.load_video(path, specific_frames=[20])
            plt.title(f'[{k}][{cluster_no}]  {filename}')
            plt.imshow(out.out_video[0])
            plt.show()

    @staticmethod
    def diff_faces(face_hashes):
        diffs = []
        base_hash = face_hashes[0]
        lengths = set([len(f) for f in face_hashes])
        if len(lengths) != 1:
            return

        for face_hash in face_hashes:
            diff_hash = ''

            for k, char in enumerate(base_hash):
                diff_hash += str(int(char == face_hash[k]))

            diffs.append(diff_hash)

        return diffs

    def manual_grade_cross(self, skip=True):
        cluster_groups = self.get_clusters_info(verbose=False)
        real, fake, mixed = cluster_groups
        clusters = {**real, **fake, **mixed}

        csv_name = 'cross-clusters-labelled.csv'
        cross_path = f'stats/bg-clusters/{csv_name}'
        cross_df = pd.read_csv(cross_path)
        indexes = cross_df.index
        k = 0

        while k < len(indexes):
            index = indexes[k]
            row = cross_df.loc[index]
            cluster_no = row['cluster']
            distance = row['nearest_distance']

            if skip and not np.isnan(row['match']):
                k += 1
                continue

            if distance == -1:
                print(f'skipping cluster {cluster_no}')
                k += 1
                continue

            cluster = clusters[cluster_no]
            cluster_filenames = cluster['filename'].to_numpy()
            cluster_labels = cluster['label'].to_numpy()
            cluster_faces = cluster['face_hash'].to_numpy()
            cluster_diffs = self.diff_faces(cluster_faces)
            name_labels = list(zip(
                cluster_filenames, cluster_labels
            ))

            nearest_clusters = row['nearest_clusters']
            nearest_files = row['nearest_files']
            nearest_files = ast.literal_eval(nearest_files)

            if len(nearest_files) < 11:
                self.thumbnail_videos(cluster_no, nearest_files)
            else:
                print(f'{len(nearest_files)} nearest filenames')

            fake_filename = cluster_filenames[0]
            file_path = f'datasets/train/videos/{fake_filename}'
            os.system(f'xdg-open {file_path}')

            print(f'cluster: {cluster_no}')
            print(f'cluster filenames: {name_labels}')
            print(f'cluster face hash: {list(cluster_faces)}')
            print(f'cluster face diff: {cluster_diffs}')

            print(f'nearest distance: {distance}')
            print(f'nearest clusters: {nearest_clusters}')
            print(f'nearest files: {nearest_files}')

            while True:
                try:
                    command = input(f'[{k}]: ').strip()
                except KeyboardInterrupt:
                    continue

                if command == 'b':
                    # print('BREAK')
                    k -= 2
                    print('BREAK', k)
                    skip = False
                    break
                elif command == 'n':
                    break

                if len(command) != 2:
                    continue

                if command[0] == 'm':
                    cross_df.loc[k, 'match'] = 1
                elif command[0] == 'u':
                    cross_df.loc[k, 'match'] = 0
                else:
                    continue

                if command[1] == 'f':
                    cross_df.loc[k, 'label'] = 'F'
                    break
                elif command[1] == 'h':
                    cross_df.loc[k, 'label'] = 'H'
                    break
                elif command[1] == 'd':
                    cross_df.loc[k, 'label'] = 'D'
                    break
                elif command[1] == 'r':
                    cross_df.loc[k, 'label'] = 'R'
                    break

            cross_df.to_csv(cross_path, index=False)
            input('>>> ')
            k += 1

    def sub_grade_clusters(self):
        cluster_groups = self.get_clusters_info(verbose=False)
        real, fake, mixed = cluster_groups
        clusters = {**real, **fake, **mixed}

        print(f'real clusters {len(real)}')
        print(f'fake clusters {len(fake)}')
        print(f'mixed clusters {len(mixed)}')

        csv_name = 'cross-clusters-labelled.csv'
        cross_path = f'stats/bg-clusters/cross-clusters-labelled.csv'
        cross_df = pd.read_csv(cross_path)
        indexes = cross_df.index
        k = 0

        half_df = cross_df[cross_df['label'] == 'H']
        half_clusters = half_df['cluster'].to_numpy()
        half_filenames, cluster_nos = [], []
        file_mapping = {}

        for cluster_no in half_clusters:
            filenames = clusters[cluster_no]['filename']
            half_filenames.extend(filenames)
            cluster_nos.extend([cluster_no] * len(filenames))

            for filename in filenames:
                file_mapping[filename] = cluster_no

        # print(mixed_filenames, len(mixed_filenames))
        out_csv_path = 'stats/bg-clusters/video-labels.csv'

        try:
            out_df = pd.read_csv(out_csv_path)
        except FileNotFoundError:
            out_df = pd.DataFrame(data={
                'filename': half_filenames,
                'cluster': cluster_nos,
                'label': ['NULL'] * len(half_filenames)
            })

        index, skip, prev_cluster_no = 0, True, None

        while index < len(half_filenames):
            filename = half_filenames[index]
            cluster_no = file_mapping[filename]
            new_cluster = cluster_no != prev_cluster_no

            row = cross_df[cross_df['cluster'] == cluster_no]
            cond = out_df['filename'] == filename
            # print(k, cluster_no, filename)

            # print(out_df[cond]['label'], out_df[cond])
            file_label = out_df[cond]['label'].to_numpy()[0]
            # print([file_label, type(file_label)])

            if skip and not (file_label is np.nan):
                index += 1
                continue

            if new_cluster:
                if prev_cluster_no is not None:
                    input('>>> ')

                prev_cluster_no = cluster_no

            cluster = clusters[cluster_no]
            cluster_filenames = cluster['filename'].to_numpy()
            cluster_labels = cluster['label'].to_numpy()
            nearest_files = row['nearest_files'].to_numpy()[0]
            nearest_files = ast.literal_eval(nearest_files)

            if new_cluster and (len(nearest_files) < 11):
                self.thumbnail_videos(cluster_no, nearest_files)
            else:
                print(f'{len(nearest_files)} nearest filenames')

            file_path = f'datasets/train/videos/{filename}'
            os.system(f'xdg-open {file_path}')

            print(f'cluster: {cluster_no}')
            print(f'nearest files: {nearest_files}')
            print(f'filename: {filename}')

            while True:
                try:
                    command = input(f'[{index}]: ').strip()
                except KeyboardInterrupt:
                    continue

                if command == 'b':
                    index -= 2
                    skip = False
                    break
                elif command == 'n':
                    break

                if command == 'f':
                    out_df.loc[cond, 'label'] = 'F'
                    break
                elif command == 'h':
                    out_df.loc[cond, 'label'] = 'H'
                    break
                elif command == 'd':
                    out_df.loc[cond, 'label'] = 'D'
                    break
                elif command == 'r':
                    out_df.loc[cond, 'label'] = 'R'
                    break

            out_df.to_csv(out_csv_path, index=False)
            index += 1

        """
        while k < len(indexes):
            index = indexes[k]
            row = cross_df.loc[index]
            cluster_no = row['cluster']
            distance = row['nearest_distance']

            if skip and not np.isnan(row['match']):
                k += 1
                continue
            elif distance == -1:
                print(f'skipping cluster {cluster_no}')
                k += 1
                continue
        """

    def auto_fill_fake_clusters(self):
        cluster_groups = self.get_clusters_info(verbose=False)
        real, fake, mixed = cluster_groups
        clusters = {**real, **fake, **mixed}

        print(f'real clusters {len(real)}')
        print(f'fake clusters {len(fake)}')
        print(f'mixed clusters {len(mixed)}')

        cross_path = f'stats/bg-clusters/cross-clusters-labelled.csv'
        cross_df = pd.read_csv(cross_path)
        indexes = cross_df.index

        base_csv_path = 'stats/bg-clusters/video-labels.csv'
        base_df = pd.read_csv(base_csv_path)
        fake_filenames = []
        cluster_log = []

        for index in tqdm(indexes):
            row = cross_df.loc[index]
            if row['label'] != 'F':
                continue

            cluster_no = row['cluster']
            cluster = clusters[cluster_no]
            filenames = cluster['filename'].to_numpy()
            cluster_log.extend([cluster_no] * len(filenames))
            fake_filenames.extend(filenames)

        fake_df = pd.DataFrame({
            'filename': fake_filenames, 'cluster': cluster_log,
            'label': ['F'] * len(fake_filenames)
        })

        out_path = f'stats/bg-clusters/video-labels-{self.stamp}.csv'
        combine_df = pd.concat([base_df, fake_df], ignore_index=True)
        combine_df.to_csv(out_path, index=False)
        print(f'wrote video labels to {out_path}')

    @staticmethod
    def get_min_distance(face_hash, real_hashes):
        min_distance = float('inf')
        face_hash = int(face_hash, 16)

        for real_hash in real_hashes:
            real_hash = int(real_hash, 16)
            mismatch = face_hash ^ real_hash
            distance = bin(mismatch).count('1')

            if distance < min_distance:
                min_distance = distance

        return min_distance

    @staticmethod
    def get_real_audio_clusters(audio_clusters, filenames):
        if type(filenames) == str:
            filenames = [filenames]

        cluster_nos = []

        for filename in filenames:
            cond = audio_clusters['filename'] == filename
            row = audio_clusters[cond]
            cluster_no = row['cluster'].to_numpy()[0]

            if cluster_no not in cluster_nos:
                cluster_nos.append(cluster_no)

        return cluster_nos

    def fill_mixed_clusters(self, save=True):
        cluster_groups = self.get_clusters_info(verbose=False)
        real, fake, mixed_clusters = cluster_groups
        # clusters = {**real, **fake, **mixed_clusters}

        audio_cluster_path = 'audio-clusters-210929-2138.csv'
        audio_cluster_path = f'stats/bg-clusters/{audio_cluster_path}'
        audio_clusters = pd.read_csv(audio_cluster_path)

        audio_path = 'stats/audio-labels-211014-1653.csv'
        audio_df = pd.read_csv(audio_path)
        audio_map = {}

        for index in audio_df.index:
            row = audio_df.loc[index]
            fake_audio, filename = row['fake_audio'], row['filename']
            audio_map[filename] = fake_audio

        print(f'real clusters {len(real)}')
        print(f'fake clusters {len(fake)}')
        print(f'mixed clusters {len(mixed_clusters)}')

        cluster_log, audio_log = [], []
        filename_log, face_label_log = [], []
        distance_log, c_fake_log = [], []
        c_real_log, label_log = [], []

        actual_reals = []
        confirm_fakes, confirm_reals = [], []
        classified_fakes, classified_reals = 0, 0

        for cluster_no in tqdm(mixed_clusters):
            cluster = mixed_clusters[cluster_no]
            # print(cluster)

            is_real = cluster['label'] == 0
            real_rows = cluster[is_real]
            # fake_rows = cluster[~is_real]

            real_hashes = real_rows['face_hash'].to_numpy()
            real_hashes = np.unique(real_hashes)
            hashes = cluster['face_hash'].to_numpy()
            labels = cluster['label'].to_numpy()
            filenames = cluster['filename'].to_numpy()
            real_filenames = real_rows['filename'].to_numpy()
            real_audio_cluster_nos = self.get_real_audio_clusters(
                audio_clusters, real_filenames
            )

            for k in range(len(filenames)):
                c_fake, c_real = 0, 0
                filename = filenames[k]
                face_hash, label = hashes[k], labels[k]
                audio_label = audio_map.get(filename, 0.5)
                distance = self.get_min_distance(
                    face_hash, real_hashes
                )

                if (label == 1) and (audio_label == 0):
                    audio_cluster_no = self.get_real_audio_clusters(
                        audio_clusters, filename
                    )

                    if audio_cluster_no in real_audio_cluster_nos:
                        confirm_fakes.append(filename)
                        face_label_log.append(1)
                    else:
                        confirm_reals.append(filename)
                        face_label_log.append(0)

                    label_log.append(label)
                    cluster_log.append(cluster_no)
                    audio_log.append(audio_label)
                    filename_log.append(filename)
                    distance_log.append(distance)

                    c_fake_log.append(0)
                    c_real_log.append(0)
                    continue

                is_fake = int(distance > 0)

                if label == 0:
                    actual_reals.append(filename)
                    assert is_fake == 0
                elif is_fake:
                    classified_fakes += 1
                    c_fake = 1
                else:
                    classified_reals += 1
                    c_real = 1

                label_log.append(label)
                cluster_log.append(cluster_no)
                audio_log.append(audio_label)
                filename_log.append(filename)
                distance_log.append(distance)
                face_label_log.append(is_fake)
                c_fake_log.append(c_fake)
                c_real_log.append(c_real)

        out_df = pd.DataFrame(data={
            'cluster': cluster_log, 'filename': filename_log,
            'fake_audio': audio_log, 'face_fake': face_label_log,
            'distance': distance_log, 'c_fake': c_fake_log,
            'c_real': c_real_log, 'label': label_log
        })

        num_fakes = face_label_log.count(1)
        num_reals = face_label_log.count(0)
        total = num_reals + num_fakes
        labelled_reals = (
            num_reals - classified_reals - len(actual_reals)
        )

        print('')
        print(f'confirm fakes: {len(confirm_fakes)}')
        print(f'classified fakes: {classified_fakes}')
        print(f'fakes: {num_fakes}')
        print('')
        print(f'confirm reals: {len(confirm_reals)}')
        print(f'actual reals: {len(actual_reals)}')
        print(f'classified reals: {classified_reals}')
        print(f'labelled reals: {labelled_reals}')
        print(f'reals: {num_reals}')
        print('')
        print(f'total: {total}')

        if save:
            name = f'mixed-fill-faces-{self.stamp}'
            out_path = f'stats/bg-clusters/{name}.csv'
            out_df.to_csv(out_path, index=False)
            print(f'saved to {out_path}')

        # print(confirm_fakes)
        return {
            'confirm_fakes': confirm_fakes,
            'actual_reals': actual_reals
        }

    def analyse_distances(self):
        name = 'mixed-fill-faces-211015-1306.csv'
        path = f'stats/bg-clusters/{name}'

        df = pd.read_csv(path)
        c_fakes = df[df['c_fake'] == 1]
        distances = c_fakes['distance'].to_numpy()
        print(len(distances))
        plt.hist(distances)
        plt.show()

    def manual_label_mixed(self):
        cluster_groups = self.get_clusters_info(verbose=False)
        real, fake, mixed = cluster_groups
        clusters = {**real, **fake, **mixed}

        audio_path = 'stats/audio-labels-211014-1653.csv'
        audio_df = pd.read_csv(audio_path)
        audio_map = {}

        for index in audio_df.index:
            row = audio_df.loc[index]
            fake_audio, filename = row['fake_audio'], row['filename']
            audio_map[filename] = fake_audio

        result = self.fill_mixed_clusters(save=False)
        confirm_fakes = result['confirm_fakes']
        actual_reals = result['actual_reals']
        ignore_files = confirm_fakes + actual_reals

        cross_path = f'stats/bg-clusters/cross-clusters-labelled.csv'
        cross_df = pd.read_csv(cross_path)
        name = 'mixed-fill-faces.csv'
        path = f'stats/bg-clusters/{name}'

        df = pd.read_csv(path)
        if 'manual' not in df:
            print('INITIALISING MANUAL')
            df['manual'] = 'X'

        rows, indexes = [], []

        for index in df.index:
            row = df.loc[index]
            filename = row['filename']
            if filename not in ignore_files:
                rows.append(row)
                indexes.append(index)

        sorted_rows = rows
        total = len(sorted_rows)
        # distances = [r['distance'] for r in rows]
        # arg_indexes = np.argsort(distances)
        # sorted_rows = [rows[index] for index in arg_indexes]
        input(f'total to fill {total}: ')
        # print(sorted_rows)

        skip = True
        prev_cluster_no = None
        completed, k = 0, 0

        while k < len(sorted_rows):
            row = sorted_rows[k]
            cluster_no = row['cluster']
            filename = row['filename']
            distance = row['distance']
            new_cluster = cluster_no != prev_cluster_no
            audio_fake = audio_map.get(filename, None)
            index = indexes[k]

            # input(f'INDEX {index}: ')
            # print(row)
            # print([k, row['manual'], index])

            if skip and (row['manual'] != 'X'):
                k += 1
                continue

            if new_cluster:
                if prev_cluster_no is not None:
                    input('>>> ')

                prev_cluster_no = cluster_no

            # print(cross_df, cluster_no)
            cluster = clusters[cluster_no]
            real_files = cluster[cluster['label'] == 0]
            nearest_files = real_files['filename'].to_numpy()

            if new_cluster and (len(nearest_files) < 11):
                self.thumbnail_videos(cluster_no, nearest_files)
            else:
                print(f'{len(nearest_files)} nearest filenames')

            file_path = f'datasets/train/videos/{filename}'
            os.system(f'xdg-open {file_path}')

            print(f'cluster: {cluster_no}')
            print(f'audio fake? {audio_fake}')
            print(f'distance: {distance}')
            print(f'nearest files: {nearest_files}')
            print(f'filename: {filename}')

            while True:
                prompt = f'[{k}/{total}][{completed}]: '

                try:
                    command = input(prompt).strip()
                except KeyboardInterrupt:
                    continue

                if command == 'b':
                    completed -= 2
                    k -= 2
                    skip = False
                    break
                elif command == 'n':
                    break

                if command == 'f':
                    df.loc[index, 'manual'] = 'F'
                    break
                elif command == 'h':
                    df.loc[index, 'manual'] = 'H'
                    break
                elif command == 'd':
                    df.loc[index, 'manual'] = 'D'
                    break
                elif command == 'r':
                    df.loc[index, 'manual'] = 'R'
                    break

            df.to_csv(path, index=False)
            completed += 1
            k += 1

    def stitch_labels(self, save=True):
        mixed_path = 'mixed-fill-faces.csv'
        mixed_path = f'stats/bg-clusters/{mixed_path}'
        half_path = 'video-labels-211015-1016.csv'
        half_path = f'stats/bg-clusters/{half_path}'
        labels_path = f'datasets/extra-labels.csv'
        cross_path = f'cross-clusters-labelled.csv'
        cross_path = f'stats/bg-clusters/{cross_path}'

        cluster_groups = self.get_clusters_info(verbose=False)
        real, fake, mixed = cluster_groups
        clusters = {**real, **fake, **mixed}

        cross_df = pd.read_csv(cross_path)
        labels_df = pd.read_csv(labels_path)
        mixed_df = pd.read_csv(mixed_path)
        half_df = pd.read_csv(half_path)
        filename_log, label_log = [], []
        cluster_log = []

        result = self.fill_mixed_clusters(save=False)
        confirm_fakes = result['confirm_fakes']
        actual_reals = result['actual_reals']

        for index in tqdm(mixed_df.index):
            row = mixed_df.loc[index]
            filename = row['filename']
            cluster_no = row['cluster']

            if filename in actual_reals:
                label = 0
            elif filename in confirm_fakes:
                label = 1
            else:
                str_label = row['manual']
                assert str_label in ('F', 'R')
                label = {'F': 1, 'R': 0}[str_label]

            cluster_log.append(cluster_no)
            filename_log.append(filename)
            label_log.append(label)

        for index in tqdm(half_df.index):
            row = half_df.loc[index]
            filename = row['filename']
            cluster_no = row['cluster']

            str_label = row['label']
            assert str_label in ('F', 'R')
            label = {'F': 1, 'R': 0}[str_label]
            cluster_log.append(cluster_no)
            assert filename not in filename_log
            filename_log.append(filename)
            label_log.append(label)

        total, skips = 0, 0
        pbar = tqdm(cross_df.index)

        for index in pbar:
            row = cross_df.loc[index]
            cluster_no = row['cluster']
            nearest_distance = row['nearest_distance']
            str_label = row['label']

            if nearest_distance == -1:
                continue
            if str_label in ('X', 'H', 'D'):
                continue

            try:
                assert str_label in ('F', 'R')
            except AssertionError as e:
                print('BAD STR LABEL', str_label)
                raise e

            label = {'F': 1, 'R': 0}[str_label]
            cluster = clusters[cluster_no]
            filenames = cluster['filename'].to_numpy()

            for filename in filenames:
                pbar.set_description(f'skip {skips}/{total}')
                total += 1

                if filename in filename_log:
                    skips += 1
                    continue

                cluster_log.append(cluster_no)
                filename_log.append(filename)
                label_log.append(label)

        for index in tqdm(labels_df.index):
            row = labels_df.loc[index]
            filename = row['filename']
            label = row['label']

            if filename in filename_log:
                continue
            if label != 0:
                continue

            cluster_log.append(-1)
            filename_log.append(filename)
            label_log.append(0)

        df = pd.DataFrame(data={
            'cluster': cluster_log, 'filename': filename_log,
            'label': label_log
        })

        if save:
            out_path = f'face-vid-labels-{self.stamp}.csv'
            out_path = f'stats/bg-clusters/{out_path}'
            df.to_csv(out_path, index=False)
            print(f'stitched labels saved to {out_path}')