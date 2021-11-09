import random

from BaseDataset import BaseDataset

class RealDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.current_mel = None
        self.current_fps = None
        self.current_image_paths = []

    def load_random_video(self):
        orig_mel, image_paths, fps = self._load_random_video()
        assert fps != 0

        self.current_fps = fps
        self.current_mel = orig_mel
        self.current_image_paths = image_paths

    def __getitem__(self, idx):
        sample = self.load_random_sample()
        return sample

    def load_random_sample(self):
        sample = None

        while sample is None:
            if len(self.current_image_paths) == 0:
                self.load_random_video()

            sample = self._load_random_sample()

        return sample

    def _load_random_sample(self):
        image_path = self.current_image_paths.pop()
        frame_no = self.get_frame_no(image_path)
        # frame_key = (filename, frame_no)

        window_fnames = self.get_window(image_path)
        if window_fnames is None:
            return None

        torch_imgs = self.batch_image_window(window_fnames)
        torch_mels = self.load_mel_batch(
            self.current_mel, self.current_fps, frame_no
        )

        if self.is_incomplete_mel(torch_mels):
            return None

        return torch_imgs, torch_mels





