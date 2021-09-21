import random
import utils

from torch.multiprocessing import Queue, Process, set_start_method

class Loader(Process):
    def __init__(
        self, samples, num_workers=4, cache_size=32,
        fetch_size=16, num_mels=16, target_lengths=(128, 128),
        fake_p=0.5
    ):
        super(Process, self).__init__()
        self.num_workers = num_workers
        self.fetch_size = fetch_size
        self.cache_size = cache_size
        self.samples = samples

        self.target_lengths = target_lengths
        self.num_mels = num_mels
        self.fake_p = fake_p

        self.train_data_queue = Queue()
        self.train_label_queue = Queue()
        self.test_data_queue = Queue()
        self.test_label_queue = Queue()

    @property
    def train_reals(self):
        return self.queues['train_real']

    @property
    def train_fakes(self):
        return self.queues['train_fake']

    @property
    def test_reals(self):
        return self.queues['test_real']

    @property
    def test_fakes(self):
        return self.queues['test_fake']

    def run(self):
        while True:
            time.sleep(0.1)

            try:
                self.loop_once()
            except Exception as e:
                print(e)
                raise e

    def loop_once(self):
        for name, group_samples in self.samples.items():
            if queue.qsize() < self.cache_size:
                samples = group_samples.get_samples(self.fetch_size)
                data_batch = self.load_batch(batch_filepaths)

    def prepare_batch(
        self, batch_size=16, fake_p=0.5, target_lengths=(128, 128),
        is_training=True
    ):
        # start = time.perf_counter()
        num_fake = int(batch_size * fake_p)
        fake_filepaths = self.get_rand_filepaths(
            1, num_fake, is_training=is_training
        )
        num_real = batch_size - num_fake
        real_filepaths = self.get_rand_filepaths(
            0, num_real, is_training=is_training
        )

        batch_filepaths = fake_filepaths + real_filepaths
        batch_labels = [1] * num_fake + [0] * num_real

        process_batch = utils.preprocess_from_filenames(
            batch_filepaths, '', batch_labels, use_parallel=True,
            num_cores=8, show_pbar=False
        )

        batch = [episode[0] for episode in process_batch]
        target_length = random.choice(range(
            target_lengths[0], target_lengths[1] + 1
        ))

        min_length = float('inf')
        for audio_arr in batch:
            min_length = min(min_length, len(audio_arr))

        data_batch = []
        min_length = min(min_length, target_length)

        for episode in batch:
            if len(episode) == min_length:
                data_batch.append(episode)
                continue

            max_start = len(episode) - min_length
            start = random.choice(range(max_start))
            clip_episode = episode[start: start + min_length]
            data_batch.append(clip_episode)

        data_batch = np.array(data_batch)
        assert data_batch.shape[2] == utils.hparams.num_mels
        batch_x = data_batch.reshape((
            len(data_batch), -1, utils.hparams.num_mels, 1
        ))

        np_labels = np.array([
            np.ones((min_length, 1)) * label
            for label in batch_labels
        ])

        # np_labels = np.expand_dims(np_labels, axis=-1)
        # end = time.perf_counter()
        # duration = end - start
        return batch_x, np_labels