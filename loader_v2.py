import cv2
import numpy as np

from loader import BoundingBox


class VideoEmpty(Exception):
    pass

class LazyVideo(object):
    def __init__(
        self, cap, every_n_frames=None, specific_frames=None,
        scale=1, max_frames=None, to_rgb=True, reset_index=True
    ):
        assert every_n_frames or specific_frames, (
            "Must supply either every n_frames or specific_frames"
        )
        assert bool(every_n_frames) != bool(specific_frames), (
            "Supply either 'every_n_frames' or 'specific_frames'"
        )

        if type(cap) == str:
            cap = cv2.VideoCapture(cap)
        elif reset_index:
            # make sure cap starts at frame -1
            cap.set(cv2.CAP_PROP_POS_FRAMES, - 1)

        self.cap = cap
        self._released = False
        self._frame_index = -1

        self.n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(f'N FRAMES IN', self.n_frames_in)

        if self.n_frames_in == 0:
            raise VideoEmpty

        if max_frames:
            self.n_frames_in = min(self.n_frames_in, max_frames)

        if every_n_frames:
            specific_frames = list(
                range(0, self.n_frames_in, every_n_frames)
            )

        self.specific_frames = specific_frames
        self.scale = scale
        self.to_rgb = to_rgb
        self.cutout = None

    def resolve_batch(self, batch):
        out_video = np.empty(
            (len(batch), self.height, self.width, 3),
            np.dtype('uint8')
        )

        for k, frame in enumerate(batch):
            assert isinstance(frame, LazyFrame)
            out_video[k] = frame.to_numpy()

        return out_video

    def auto_resize_inplace(self, *args, **kwargs):
        # resolution = (self.width, self.height)
        self.cutout = self.cut_blackout(
            self.out_video, *args, **kwargs
        )

    def __len__(self):
        return len(self.specific_frames)

    def __getitem__(self, index):
        frame_no = self.specific_frames[index]
        return LazyFrame(frame_no, video=self)

    @property
    def resolution(self):
        return self.width, self.height

    def load_frame_by_index(self, index):
        frame_no = self.specific_frames[index]
        return self.load_frame(frame_no)

    def load_frame(self, frame_no):
        frame = self.grab_raw_frame(frame_no)
        # print('FRAME', frame.shape, self.resolution)

        if self.scale:
            frame = cv2.resize(frame, self.resolution)
        if self.to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.cutout is not None:
            x_start, x_end, y_start, y_end = self.cutout.to_tuple()
            frame = frame[y_start:y_end, x_start:x_end]
            frame = cv2.resize(frame, self.resolution)

        return frame

    def grab_raw_frame(self, frame_no):
        if self._frame_index >= frame_no:
            # if current frame index is more than frame_no
            # we have to set the capture frame no (very slow)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
            self._frame_index = frame_no - 1

        assert frame_no > self._frame_index
        while frame_no - self._frame_index > 1:
            # grab skips ahead by 1 frame
            self.cap.grab()
            self._frame_index += 1

        # read skips ahead 1 frame, and reads it
        res, frame = self.cap.read()
        self._frame_index += 1
        assert self._frame_index == frame_no
        return frame

    @property
    def width(self):
        return int(self.width_in * self.scale)

    @property
    def height(self):
        return int(self.height_in * self.scale)

    @property
    def released(self):
        return self._released

    @property
    def out_video(self):
        return self

    def release(self):
        if not self.released:
            self.cap.release()
            self._released = True

        assert self._released
        return self.released

    @staticmethod
    def stripe_v2(
        gray_frame, size, scan_x=True,
        backwards=False
    ):
        index = 0

        for index in range(size):
            position = -index if backwards else index

            if scan_x:
                # get vertical stripe
                stripe = gray_frame[:, position]
            else:
                # get horizontal stripe
                stripe = gray_frame[position, :]

            stripe = stripe.flatten()
            p_black = sum(stripe > 2) / len(stripe)

            if p_black > 0.01:
                break

        return index

    @classmethod
    def cut_blackout(cls, images=None, samples=1):
        sample_interval = len(images) // (samples + 2)
        x_starts, y_starts = [], []
        x_ends, y_ends = [], []

        for k in range(samples):
            interval = (sample_interval + 1) * k
            frame = images[interval]
            if isinstance(frame, LazyFrame):
                frame = frame.to_numpy()

            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                assert len(frame.shape) == 2
                gray_frame = frame

            width, height = frame.shape[1], frame.shape[0]

            x_start = cls.stripe_v2(
                gray_frame, width, scan_x=True, backwards=False
            )
            x_back_end = cls.stripe_v2(
                gray_frame, width, scan_x=True, backwards=True
            )
            y_start = cls.stripe_v2(
                gray_frame, height, scan_x=False, backwards=False
            )
            y_back_end = cls.stripe_v2(
                gray_frame, height, scan_x=False, backwards=True
            )

            x_starts.append(x_start)
            x_ends.append(width - x_back_end)
            y_starts.append(y_start)
            y_ends.append(height - y_back_end)

        x_start = int(np.min(x_starts))
        x_end = int(np.max(x_ends))
        y_start = int(np.min(y_starts))
        y_end = int(np.max(y_ends))

        return BoundingBox(x_start, x_end, y_start, y_end)


class LazyFrame(object):
    def __init__(self, frame_no: int, video: LazyVideo):
        self.video = video
        self.frame_no = frame_no

    def to_numpy(self):
        return self.video.load_frame(self.frame_no)


def load_video(*args, **kwargs):
    try:
        return LazyVideo(*args, **kwargs)
    except VideoEmpty:
        return None