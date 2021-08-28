import matplotlib
import torchvision.transforms as transforms
import pandas as pd
import sklearn
import loader
import face_recognition
import cv2

from PIL import Image
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(
        self, seed=42, train_size=0.9, basedir='datasets'
    ):
        self.basedir = basedir
        self.csv_filename = f'{basedir}/train.csv'
        self.labels = pd.read_csv(self.csv_filename)
        self.dir_mapping = {}
        self.seed = seed

        video_names = self.labels['filename']
        split = train_test_split(
            video_names, random_state=42, train_size=train_size
        )

        train, test = split
        self.all_videos = video_names
        self.train_videos = train.to_numpy()
        self.test_videos = test.to_numpy()
        print('TRAIN SIZE', len(train))
        print('TEXT SIZE', len(test))

    def get_face_locations(
        self, filename, every_n_frames, rescale,
        return_vid_obj=True
    ):
        video_faces = {}
        vid_obj = self.read_video(
            filename, every_n_frames=every_n_frames,
            rescale=rescale
        )

        np_frames = vid_obj.out_video
        num_frames = len(np_frames)

        for i in range(num_frames):
            image = np_frames[i]
            frame_no = every_n_frames * i
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) > 0:
                video_faces[frame_no] = face_locations

        vid_obj.video_faces = video_faces

        if return_vid_obj:
            return vid_obj
        else:
            return video_faces

    def fetch_row(self, filename: str):
        assert type(filename) is str
        row = self.labels[self.labels['filename'] == filename]
        return row

    def load_video(self, filename):
        path = f'{self.basedir}/train/videos/{filename}'
        video = cv2.VideoCapture(path)
        return video

    def read_video(self, filename, *args, **kwargs):
        video_capture = self.load_video(filename)
        video_arr = loader.load_video(
            video_capture, filename=filename, *args, **kwargs
        )
        return video_arr

    def is_fake(self, filename: str):
        row = self.fetch_row(filename)
        is_fake = row['label'].to_numpy()[0]
        return is_fake


if __name__ == '__main__':
    dataset = Dataset()
    print('example train videos: ', dataset.train_videos[:10])
    print('example test videos: ', dataset.test_videos[:10])