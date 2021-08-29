# import ParentImport
import datasets
import torchvision.transforms as transforms
import face_recognition
import pandas as pd
import cProfile
import datetime
import torch
import json

from tqdm.auto import tqdm
from matplotlib.pyplot import imshow
# from trainer import Trainer

dt = datetime.datetime.now()
stamp = dt.strftime('%Y%m%d-%H%M%S')
dataset = datasets.Dataset(basedir='datasets')
# dataset.all_videos

dataframe = pd.DataFrame(columns=['filename', 'max_faces'])
face_coords_df = pd.DataFrame(columns=[
    'filename', 'frames', 'face_no', 'num_faces',
    'top', 'right', 'bottom', 'left'
])

print(dataframe.head())
print(len(dataset.train_videos), dataset.train_videos[:5])
print(len(dataset.test_videos), dataset.test_videos[:5])

max_face_mapping = {}
length = len(dataset.all_videos)

scale_down = 4
every_n_frames = 20
n = every_n_frames

assert type(scale_down) is int
assert scale_down >= 1
rescale = 1 / scale_down

json_filename = f'stats/face_map_stats-{scale_down}-{n}-{stamp}.json'
csv_filename = f'stats/face_map_stats-{scale_down}-{n}-{stamp}.csv'
profile_path = f'stats/profile-{scale_down}-{n}-{stamp}.profile'

print(f'rescale is {rescale}')
print(f'JSON filename is {json_filename}')
print(f'CSV filename is {csv_filename}')

def fill_face_maps(np_frames, interval):
    face_mapping = {}
    max_faces = 0

    for i in range(len(np_frames)):
        image = np_frames[i]
        frame_no = interval * i
        face_locations = face_recognition.face_locations(image)
        face_mapping[frame_no] = face_locations

        num_faces = len(face_locations)
        # print('faces', num_faces, max_faces)
        max_faces = max(max_faces, num_faces)

    return max_faces, face_mapping


def load_video(filename, scale, n_frames):
    try:
        vid_obj = dataset.read_video(
            filename, rescale=scale,
            every_n_frames=n_frames
        )
    except ValueError as e:
        print('FAILED TO READ', filename)
        return

    np_frames = vid_obj.out_video
    return np_frames


def conditional_load(filename, scale, n_frames):
    np_frames = load_video(filename, scale, n_frames)
    if np_frames is None:
        return None

    num_frames = len(np_frames)
    max_faces, face_mapping = fill_face_maps(
        np_frames, every_n_frames
    )

    return (
        num_frames, np_frames, max_faces,
        face_mapping
    )

def run():
    column = 0

    for k in tqdm(range(length)):
        filename = dataset.all_videos[k]
        load_result = conditional_load(
            filename, scale=rescale, n_frames=every_n_frames
        )

        if load_result is None:
            continue

        current_rescale = rescale
        num_frames, np_frames, max_faces, face_mapping = load_result

        if max_faces == 0:
            print(f'NO FACES DETECTED [{filename}] - UNSCALING')
            current_rescale = 1

            load_result = conditional_load(
                filename, scale=current_rescale,
                n_frames=every_n_frames
            )

        num_frames, np_frames, max_faces, face_mapping = load_result
        detections = 0

        for frame_no in face_mapping:
            image = np_frames[frame_no // every_n_frames]
            face_locations = face_recognition.face_locations(image)
            faces = len(face_locations)

            if faces > 0:
                detections += 1

            max_faces = max(max_faces, faces)
            num_faces = len(face_locations)

            for face_no in range(len(face_locations)):
                face_location = face_locations[face_no]
                top, right, bottom, left = face_location

                top = int(top / current_rescale)
                right = int(right / current_rescale)
                bottom = int(bottom / current_rescale)
                left = int(left / current_rescale)

                face_coords_df.loc[column] = [
                    filename, frame_no, face_no, num_faces,
                    top, right, bottom, left
                ]

                column += 1
                # print(face_coords_df)
                # input('>>> ')

        p_detection = detections / num_frames
        max_face_mapping[filename] = (max_faces, p_detection)
        print(f'READ {k}/{length}', filename, max_faces, p_detection)

    face_coords_df.to_csv(csv_filename)

    with open(json_filename, 'w') as fp:
        json.dump(max_face_mapping, fp, indent=4)


profile = cProfile.Profile()
profile.enable()
run()
profile.disable()
profile.dump_stats(profile_path)