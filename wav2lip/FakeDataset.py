import torch
import random
import numpy as np

try:
    from BaseDataset import BaseDataset
except ModuleNotFoundError:
    from .BaseDataset import BaseDataset

class FakeDataset(BaseDataset):
    def __init__(self, *args, cache_size=4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = 'F'

        self.mel_stack = []
        self.img_stack = []

        self.cache_size = cache_size
        self.fake_img_cache = [None] * cache_size
        self.fake_mel_cache = [None] * cache_size
        # self.pre_populate()

    def __iter__(self):
        while True:
            sample = self._load_random_sample()
            yield sample

    def __getitem__(self, idx):
        sample = self._load_random_sample()
        return sample

    def load_random_video(self):
        name = self.choose_random_name()
        filename = f'{name}.mp4'
        # print('FILENAME', filename)

        assert type(filename) is str
        current_mel = self.load_audio(filename)
        current_fps = self.resolve_fps(filename)
        assert current_fps != 0

        img_samples, mel_samples = [], []
        image_paths = self.load_image_paths(
            name, randomize_images=True
        )

        for image_path in image_paths:
            frame_no = self.get_frame_no(image_path)
            window_fnames = self.get_window(image_path)
            torch_mels = self.load_random_torch_mel(
                frame_no, current_mel, current_fps
            )

            if window_fnames is None:
                continue

            torch_imgs = self.batch_image_window(window_fnames)
            img_samples.append(torch_imgs)
            mel_samples.append(torch_mels)

        assert len(img_samples) == len(mel_samples)
        self.img_stack.extend(img_samples)
        self.mel_stack.extend(mel_samples)

    def load_random_torch_mel(
        self, exclude_frame_no, current_mel, current_fps
    ):
        max_audio_frame = -1 + self.get_audio_max_frame(
            current_mel, current_fps
        )
        while True:
            aud_frame_no = random.choice(range(max_audio_frame))
            if aud_frame_no == exclude_frame_no:
                continue

            torch_mels = self.load_mel_batch(
                current_mel, current_fps, aud_frame_no
            )
            if self.is_complete_mel(torch_mels):
                return torch_mels

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
        print(f'IMG-STACK [{self.name}] {len(self.img_stack)}')
        print(f'MEL-STACK [{self.name}] {len(self.mel_stack)}')

    def pre_populate(self):
        while len(self.img_stack) < self.cache_size:
            self.load_random_video()
            self.log_stack_status()

        self.verify_stacks()
        for k in range(self.cache_size):
            img_sample = self.img_stack.pop()
            mel_sample = self.mel_stack.pop()
            self.fake_img_cache[k] = img_sample
            self.fake_mel_cache[k] = mel_sample

    def _choose_sample(self, filename):
        assert type(filename) is str
        current_fps = self.resolve_fps(filename)
        assert current_fps != 0

        name = filename[:filename.rindex('.')]
        frame_no, window_fnames = 0, None
        image_paths = self.load_image_paths(
            name, randomize_images=True
        )

        while window_fnames is None:
            image_path = random.choice(image_paths)
            frame_no = self.get_frame_no(image_path)
            window_fnames = self.get_window(image_path)

        torch_imgs = self.batch_image_window(window_fnames)
        torch_mels = self._load_random_audio_sample(
            filename, exclude_frame_no=frame_no
        )

        return torch_imgs, torch_mels

    def _load_random_sample(self):
        return self.choose_sample()

    def choose_sample(self, filename=None):
        if filename is None:
            name = self.choose_random_name()
            filename = f'{name}.mp4'

        # print('FILENAME', filename)
        return self._choose_sample(filename)

    def _load_random_audio_sample(
        self, exclude_filename, exclude_frame_no
    ):
        filename = self.choose_random_filename()
        if filename != exclude_filename:
            exclude_frame_no = None

        current_mel = self.load_audio(filename)
        current_fps = self.resolve_fps(filename)

        assert current_fps != 0
        torch_mels = self.load_random_torch_mel(
            exclude_frame_no, current_mel, current_fps
        )

        return torch_mels

    def _load_random_sample_v1(self):
        while len(self.img_stack) == 0:
            self.load_random_video()

        self.verify_stacks()
        index = random.choice(range(self.cache_size))
        img_sample = self.fake_img_cache[index]
        mel_sample = self.fake_mel_cache[index]
        assert (img_sample is not None) and (mel_sample is not None)

        self.fake_img_cache[index] = self.img_stack.pop()
        self.fake_mel_cache[index] = self.mel_stack.pop()
        return img_sample, mel_sample

    def verify_stacks(self):
        assert len(self.mel_stack) == len(self.img_stack)