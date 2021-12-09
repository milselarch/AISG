import math

class FaceImageMap(object):
    def __init__(self, face_image_map, fps: int):
        self.face_image_map = face_image_map
        self.fps = fps

    def __iter__(self):
        for key in self.face_image_map:
            yield key

    def __getitem__(self, face_no):
        return self.get_face_frames(face_no)

    def get_face_frames(self, face_no):
        return self.face_image_map[face_no]

    def __len__(self):
        return len(self.face_image_map)

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}({repr(self.face_image_map)})'

    @property
    def face_nos(self):
        return list(self.face_image_map.keys())

    def get_detected_frames(self, face_no):
        face_frames = self.get_face_frames(face_no)
        face_samples = {}

        for frame_no in face_frames:
            face_image = face_frames[frame_no]
            if face_image.detected:
                face_samples[frame_no] = face_image

        return face_samples

    def sample_detected_frames(
        self, face_no, max_samples=64, min_samples=20,
        clip_start=True, clip_end=True
    ):
        face_samples = self.get_detected_frames(face_no)
        frame_nos = list(face_samples.keys())
        print('FACE SAMPLES', len(frame_nos))

        if clip_start and len(frame_nos) > min_samples:
            frame_nos = frame_nos[1:]
        if clip_end and len(frame_nos) > min_samples:
            frame_nos = frame_nos[:-1]

        new_face_samples = {}
        samples_taken, prev_frame_no = 0, -1
        interval = math.ceil(len(frame_nos) / max_samples)
        interval = max(interval, 1)

        for k in range(len (frame_nos)):
            sample_progress = k / interval
            frame_no = frame_nos[k]

            if sample_progress < samples_taken:
                continue
            if prev_frame_no == frame_no:
                continue

            new_face_samples[frame_no] = face_samples[frame_no]
            prev_frame_no = frame_no
            samples_taken += 1

        try:
            assert len(new_face_samples) <= max_samples
        except AssertionError as e:
            print('FACE SAMPLES TOO MANY', len(new_face_samples))
            raise e

        return new_face_samples

    def sample_face_frames(
        self, face_no, consecutive_frames=1, max_samples=32,
        require_first_detected=True, extract=False, clip_p=0,
        min_samples=20, clip_start=True, clip_end=True
    ):
        if max_samples is None:
            max_samples = float('inf')

        face_frames = self.get_face_frames(face_no)
        allowed_frame_nos = []

        for frame_no in face_frames:
            face_image = face_frames[frame_no]
            if require_first_detected and not face_image.detected:
                continue

            has_consecutive = True
            for k in range(consecutive_frames):
                future_frame_no = frame_no + k
                if future_frame_no not in face_frames:
                    has_consecutive = False
                    break

            if has_consecutive:
                allowed_frame_nos.append(frame_no)

        if clip_start and len(allowed_frame_nos) > min_samples:
            allowed_frame_nos = allowed_frame_nos[1:]
        if clip_end and len(allowed_frame_nos) > min_samples:
            allowed_frame_nos = allowed_frame_nos[:-1]

        clip = int(len(allowed_frame_nos) * clip_p)
        max_clip = (len(allowed_frame_nos) - min_samples) // 2
        max_clip = max(max_clip, 0)
        clip = min(clip, max_clip)

        end_index = len(allowed_frame_nos) - clip
        allowed_frame_nos = allowed_frame_nos[clip:end_index]
        interval = math.ceil(len(allowed_frame_nos) / max_samples)
        interval = max(interval, 1)
        samples_taken, prev_frame_no = 0, -1
        samples = []

        for k in range(len(allowed_frame_nos)):
            sample_progress = k / interval
            if sample_progress < samples_taken:
                continue

            face_samples = []
            frame_no = allowed_frame_nos[k]
            if prev_frame_no == frame_no:
                continue

            for i in range(consecutive_frames):
                window_frame_no = frame_no + i
                face_image = face_frames[window_frame_no]
                if extract:
                    face_image = face_image.face_crop

                face_samples.append(face_image)

            prev_frame_no = frame_no
            samples.append(face_samples)
            samples_taken += 1

        assert len(samples) <= max_samples
        return samples



