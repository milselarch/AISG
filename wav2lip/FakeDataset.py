import random

from BaseDataset import BaseDataset

class FakeDataset(BaseDataset):
    def __init__(self, *args, cache_size=1000, **kwargs):
        super().__init__(*args, **kwargs)

        self.mel_stack = []
        self.img_stack = []

        self.img_cache = [None] * cache_size
        self.mel_cache = [None] * cache_size
        self.cache_size = cache_size
        self.pre_populate()

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
        indexes = random.sample(
            range(max_start_index), k=max_start_index
        )

        for start_index in indexes:
            sub_mel = self.crop_audio_by_index(orig_mel, start_index)
            torch_mels = torch.FloatTensor(sub_mel.T)
            torch_mel_sample = torch_mels.unsqueeze(0).unsqueeze(0)
            if self.is_incomplete_mel(torch_mel_sample):
                continue

            self.mel_stack.append(torch_mel_sample)

    def log_cache_status(self):
        print(f'IMG-CACHE {len(self.img_cache)}')
        print(f'MEL-CACHE {len(self.mel_cache)}')

    def pre_populate(self):
        while len(self.img_stack) < self.cache_size:
            self.log_cache_status()
            self.load_random_video()
        while len(self.mel_stack) < self.cache_size:
            self.log_cache_status()
            self.load_random_audio()

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