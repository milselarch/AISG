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

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}({repr(self.face_image_map)})'

    def get_detected_frames(self, face_no):
        face_frames = self.get_face_frames(face_no)
        face_samples = []

        for frame_no in face_frames:
            face_image = face_frames[frame_no]
            if face_image.detected:
                face_samples.append(face_image)

        return face_samples

    def sample_face_frames(
        self, face_no, consecutive_frames=1, max_samples=32,
        require_first_detected=True, extract=False
    ):
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

        interval = len(allowed_frame_nos) / max_samples
        samples_taken = 0
        samples = []

        for k in range(len(allowed_frame_nos)):
            sample_progress = k / interval
            if sample_progress < samples_taken:
                continue

            face_samples = []
            frame_no = allowed_frame_nos[k]

            for i in range(consecutive_frames):
                future_frame_no = frame_no + i
                face_image = face_frames[future_frame_no]
                if extract:
                    face_image = face_image.face_crop

                face_samples.append(face_image)

            samples.append(face_samples)
            samples_taken += 1

        assert len(samples) <= max_samples
        return samples



