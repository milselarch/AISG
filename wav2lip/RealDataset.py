import random

from BaseDataset import BaseDataset
from queue import Empty as QueueEmpty

class RealDataset(BaseDataset):
    def __init__(
        self, *args, queue=None, num_caches=256,
        cache_all=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tag = 'R'

        self.queue = queue
        if self.queue is not None:
            try:
                self.ID = self.queue.get_nowait()
                print(f'QID {self.ID}')
            except QueueEmpty:
                pass

        self.cache_all = cache_all
        self.num_caches = num_caches
        self.rand_cache = {
            k: self.make_cache() for k in range(self.num_caches)
        }

    @staticmethod
    def make_cache():
        return {'mel': None, 'fps': None, 'image_paths': []}

    def load_random_video(self, cache):
        orig_mel, image_paths, fps = self._load_random_video()

        cache['fps'] = fps
        cache['mel'] = orig_mel
        cache['image_paths'] = image_paths

    def __getitem__(self, idx):
        sample = self.load_sample()
        return sample

    def __iter__(self):
        while True:
            sample = self.load_sample()
            yield sample

    def print_status(self):
        if self.cache_all:
            filled = len(self.mel_cache)
            length = len(self.file_map)
            print(f'MEL-L [{self.name}] {filled}/{length}')
        else:
            filled = sum([
                int(len(cache['image_paths']) > 0)
                for cache in self.rand_cache.values()
            ])

            length = len(self.rand_cache)
            print(f'IMG-L [{self.name}] {filled}/{length}')

    def load_sample(self):
        if not self.cache_all:
            return self.load_random_sample()
        else:
            return self.load_deliberate_sample()

    def load_deliberate_sample(self, filename=None):
        if filename is None:
            filename = self.choose_random_filename()

        current_mel = self.load_audio(filename)
        current_fps = self.resolve_fps(filename)

        while True:
            image_path, torch_imgs = self.load_torch_window(filename)
            frame_no = self.get_frame_no(image_path)
            torch_mels = self.load_mel_batch(
                current_mel, current_fps, frame_no
            )

            if self.is_incomplete_mel(torch_mels):
                continue

            return torch_imgs, torch_mels

    def load_random_sample(self):
        sample = None

        while sample is None:
            cache = self.choose_random_cache()
            current_image_paths = cache['image_paths']

            if len(current_image_paths) == 0:
                self.load_random_video(cache)
                self.print_status()

            sample = self._load_random_sample(cache)

        return sample

    def choose_random_cache(self):
        index = random.choice(range(self.num_caches))
        cache = self.rand_cache[index]
        return cache

    def load_torch_window(self, filename):
        window_fnames, image_path = None, None

        while window_fnames is None:
            image_path = self.load_image_paths(
                filename, randomize_images=True, num_samples=1
            )[0]

            window_fnames = self.get_window(image_path)

        torch_imgs = self.batch_image_window(window_fnames)
        return image_path, torch_imgs

    def _load_random_sample(self, cache):
        current_mel = cache['mel']
        current_fps = cache['fps']

        image_path = cache['image_paths'].pop()
        frame_no = self.get_frame_no(image_path)
        # frame_key = (filename, frame_no)

        window_fnames = self.get_window(image_path)
        if window_fnames is None:
            return None

        torch_imgs = self.batch_image_window(window_fnames)
        torch_mels = self.load_mel_batch(
            current_mel, current_fps, frame_no
        )

        if self.is_incomplete_mel(torch_mels):
            return None

        return torch_imgs, torch_mels





