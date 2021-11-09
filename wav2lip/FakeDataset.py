import torch
import random
import numpy as np

from BaseDataset import BaseDataset

class FakeDataset(BaseDataset):
    def __init__(self, *args, cache_size=1000, **kwargs):
        super().__init__(*args, **kwargs)

        self.mel_stack = []
        self.img_stack = []

        self.cache_size = cache_size
        self.img_cache = [None] * cache_size
        self.mel_cache = [None] * cache_size
        self.pre_populate()

    def __iter__(self):
        while True:
            sample = self._load_random_sample()
            yield sample

    def __getitem__(self, idx):
        sample = self._load_random_sample()
        return sample

    def load_random_video(self):
        img_filename = self.choose_random_filename()
        image_paths = self.load_image_paths(
            img_filename, randomize_images=True
        )

        for image_path in image_paths:
            window_fnames = self.get_window(image_path)
            if window_fnames is None:
                continue

            torch_imgs = self.batch_image_window(window_fnames)
            self.img_stack.append(torch_imgs)

    def load_random_audio(self):
        audio_filename = self.choose_random_filename()
        orig_mel = self.load_audio(audio_filename)
        max_start_index = len(orig_mel) - self.syncnet_mel_step_size

        indexes = np.arange(max_start_index)
        indexes = np.random.choice(indexes, size=len(indexes))
        indexes = indexes[::self.syncnet_mel_step_size]

        for start_index in indexes:
            sub_mel = self.crop_audio_by_index(orig_mel, start_index)
            torch_mels = torch.FloatTensor(sub_mel.T)
            torch_mel_sample = torch_mels.unsqueeze(0).unsqueeze(0)
            if self.is_incomplete_mel(torch_mel_sample):
                continue

            self.mel_stack.append(torch_mel_sample)

    def log_stack_status(self):
        print(f'IMG-STACK [{self.ID}] {len(self.img_stack)}')
        print(f'MEL-STACK [{self.ID}] {len(self.mel_stack)}')

    def pre_populate(self):
        while len(self.img_stack) < self.cache_size:
            self.load_random_video()
            self.log_stack_status()
        while len(self.mel_stack) < self.cache_size:
            self.load_random_audio()
            self.log_stack_status()

        for k in range(self.cache_size):
            img_sample = self.img_stack.pop()
            mel_sample = self.mel_stack.pop()
            self.img_cache[k] = img_sample
            self.mel_cache[k] = mel_sample

    def _load_random_sample(self):
        while len(self.img_stack) == 0:
            self.load_random_video()
        while len(self.mel_stack) == 0:
            self.load_random_audio()

        index = random.choice(range(self.cache_size))
        img_sample = self.img_cache[index]
        mel_sample = self.mel_cache[index]
        assert (img_sample is not None) and (mel_sample is not None)

        self.img_cache[index] = self.img_stack.pop()
        self.mel_cache[index] = self.mel_stack.pop()
        return img_sample, mel_sample