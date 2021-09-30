import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import hparams
from trainer import Trainer
from tqdm.auto import tqdm
from scipy import stats


class AudioAnalysis(object):
    def __init__(
        self, labels_path='../datasets/extra-labels.csv',
        audio_dir='../datasets-local/audios-flac', audio_ext='flac',
        durations_path = 'csvs/aisg-durations-210929-0931.csv',
        clip_start=160, clip_length=80
    ):
        self.cache = {}
        self.audio_dir = audio_dir
        self.audio_ext = audio_ext
        self.clip_start = clip_start
        self.clip_length = clip_length

        self.labels_path = labels_path
        self.durations_path = durations_path
        self.stamp = Trainer.make_date_stamp()

        self.duration_map = None

    def resolve(self, filename, cache_mel=True):
        try:
            name = filename[:filename.index('.')]
        except ValueError as e:
            print('bad filename', filename)
            raise e

        file_path = f'{self.audio_dir}/{name}.{self.audio_ext}'

        if not os.path.exists(file_path):
            invalid.append(file_path)
            raise FileNotFoundError

        if name in self.cache:
            return self.cache[name]

        mel, _, duration = utils.process(file_path, '', 0)
        clip_end = self.clip_start+self.clip_length
        mel = mel[self.clip_start: clip_end]

        if cache_mel:
            self.cache[name] = mel

        return mel

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
            file_path = f'../datasets-local/audios-flac/{name}.flac'

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

    def clear_cache(self):
        self.cache = {}

    def get_cluster_main(self, df, cluster_no):
        cluster = df[df['cluster'] == cluster_no]
        mode = min(stats.mode(cluster['distance']).mode)
        target = cluster[cluster['distance'] == mode]
        filename = target['filename'].to_numpy()[0]
        mel = self.resolve(filename)
        return mel, filename

    def make_clusters(self, files, pbar, threshold):
        self.clear_cache()
        clusters = []
        distances = []

        while len(files) > 0:
            new_files = []
            base_filename = files[0]
            base_mel = self.resolve(base_filename)
            cluster_dist = [0.0]
            cluster = [base_filename]
            pbar.update()

            for filename in files[1:]:
                mel = self.resolve(filename)
                distance = np.linalg.norm(mel - base_mel)

                if distance > threshold:
                    new_files.append(filename)
                else:
                    cluster.append(filename)
                    cluster_dist.append(distance)
                    pbar.update()

            clusters.append(cluster)
            distances.append(cluster_dist)
            # print(base_filename, files[1:])
            # print(cluster_dist)

            files = new_files

        # print(clusters, distances)
        return clusters, distances

    def cluster(self, threshold=17, max_durations=None):
        duration_map = self.build_duration_map()
        num_files = sum([
            len(duration_map[duration]['real']) +
            len(duration_map[duration]['fake'])
            for duration in duration_map
        ])

        cluster_no = 0
        cluster_log, label_log = [], []
        distance_log, filename_log = [], []

        pbar = tqdm(range(num_files))
        durations = list(duration_map.keys())
        if max_durations is not None:
            durations = durations[:max_durations]

        for duration in durations:
            mapping = duration_map[duration]
            reals = mapping['real']
            fakes = mapping['fake']
            files = reals + fakes

            clusters, distances = self.make_clusters(
                files, pbar, threshold=threshold
            )

            # print('CLUSTERS', clusters)
            # print('DISTANCES', distances)

            for k, cluster in enumerate(clusters):
                dist_batch = distances[k]
                assert len(dist_batch) == len(cluster)

                for i, filename in enumerate(cluster):
                    label = 0 if filename in reals else 1
                    distance = dist_batch[i]

                    filename_log.append(filename)
                    distance_log.append(distance)
                    cluster_log.append(cluster_no)
                    label_log.append(label)

                cluster_no += 1
                desc = f'clusters: {cluster_no}'
                pbar.set_description(desc)

        cluster_df = pd.DataFrame(data={
            'cluster': cluster_log, 'filename': filename_log,
            'distance': distance_log, 'label': label_log
        })

        path = f'csvs/clusters-{self.stamp}.csv'
        cluster_df.to_csv(path, index=False)
        print(f'clusters saved at {path}')
        return cluster_df

    def evaluate_clusters(self):
        path = 'csvs/clusters-210929-2138.csv'
        df = pd.read_csv(path)
        print(df)

        clusters = np.unique(df['cluster'].to_numpy())
        real_clusters, fake_clusters = [], []

        for cluster_no in clusters:
            cluster = df[df['cluster'] == cluster_no]
            is_fake = min(cluster['label']) == 1
            is_real = max(cluster['label']) == 0

            if is_fake:
                fake_clusters.append(cluster_no)
            else:
                real_clusters.append(cluster_no)

            """
            if is_real:
                real_clusters.append(cluster_no)
            elif is_fake:
                fake_clusters.append(cluster_no)
            """

        print(f'real clusters: {len(real_clusters)}')
        print(f'fake clusters: {len(fake_clusters)}')

        distances = []
        fake_filenames, real_filenames = [], []
        pairs = len(real_clusters) * len(fake_clusters)
        pbar = tqdm(range(pairs))

        for fake_cluster_no in fake_clusters:
            fake_rep = self.get_cluster_main(df, fake_cluster_no)
            fake_mel, fake_file = fake_rep
            min_distance = float('inf')
            nearest = None

            for real_cluster_no in real_clusters:
                real_rep = self.get_cluster_main(df, real_cluster_no)
                real_mel, real_file = real_rep

                distance = np.linalg.norm(fake_mel - real_mel)
                if distance < min_distance:
                    min_distance = distance
                    nearest = real_file

                dist_str = str(round(distance, 1))
                min_dist_str = str(round(min_distance, 1))
                desc = (
                    f'{fake_file}, {real_file}, {nearest} - ' +
                    f'{dist_str}, {min_dist_str}'
                )

                pbar.set_description(desc)
                pbar.update(1)

            fake_filenames.append(fake_file)
            real_filenames.append(nearest)
            distances.append(min_distance)

        dataframe = pd.DataFrame(data={
            'fake': fake_filenames, 'real': real_filenames,
            'distance': distances
        })

        path = f'csvs/cross-{self.stamp}.csv'
        dataframe.to_csv(path, index=False)
        print(f'cross-compare saved to {path}')
        print('END')

    def evaluate_cross_cmp(self):
        df = pd.read_csv('csvs/cross-210930-0203.csv')
        plt.hist(df['distance'], bins=30)
        plt.show()

    def manual_cross_cmp(self):
        path = 'csvs/cross-extra-210930-1006.csv'
        df = pd.read_csv(path)
        fakes = df['fake'].to_numpy().tolist()
        label = 0
        index = -1

        while True:
            index += 1
            fake_filename = fakes[index]
            cond = df['fake'] == fake_filename
            label = df[cond]['fake_audio'].to_numpy()[0]

            if not np.isnan(label):
                print(df[cond])
            else:
                break

        while index < len(fakes):
            fake_filename = fakes[index]
            file_path = f'../datasets/train/videos/{fake_filename}'
            os.system(f'xdg-open {file_path}')
            cond = df['fake'] == fake_filename

            row = df[cond]
            label = df[cond]['fake_audio'].to_numpy()[0]
            if not np.isnan(label):
                print(f'LABEL', label)

            print(row)

            valid = False
            stop = False

            while not valid:
                valid = True

                while True:
                    try:
                        ans = input(f'[{index}]: ').strip()
                        break
                    except KeyboardInterrupt:
                        pass

                if ans == 'u':
                    index -= 2
                elif ans == 'q':
                    stop = True
                elif ans == 'r':
                    df.loc[cond, 'fake_audio'] = 0
                    df.to_csv(path, index=False)
                    print('updated to 0')
                elif ans == 'f':
                    df.loc[cond, 'fake_audio'] = 1
                    df.to_csv(path, index=False)
                    print('updated to 1')
                elif ans == 'h':
                    df.loc[cond, 'fake_audio'] = 0.5
                    df.to_csv(path, index=False)
                    print('updated to 0.5')
                elif ans == 'n':
                    break
                elif ans.startswith('goto '):
                    try:
                        index = int(ans[5:]) - 1
                    except TypeError:
                        valid = False
                else:
                    valid = False

            if stop:
                break

            index += 1

        df.to_csv(path, index=False)
        print(f'saved to {path}')


if __name__ == '__main__':
    analyser = AudioAnalysis()
    analyser.manual_cross_cmp()