# import ParentImport
import datasets
import torchvision.transforms as transforms
import face_recognition
import pandas as pd
import torch
import json

from tqdm.auto import tqdm
from matplotlib.pyplot import imshow
# from trainer import Trainer

dataset = datasets.Dataset(basedir='datasets')
# dataset.all_videos

dataframe = pd.DataFrame(columns=['filename', 'max_faces'])
print(dataframe.head())
print(len(dataset.train_videos), dataset.train_videos[:5])
print(len(dataset.test_videos), dataset.test_videos[:5])
vid_obj = dataset.read_video(
    '9bc513e4e366b7d8.mp4', every_n_frames=20, rescale=0.1
)

image = vid_obj.out_video[9]
print(image.shape)
face_locations = face_recognition.face_locations(image)
print(face_locations)
loc1 = face_locations[0]

top, right, bottom, left = loc1
# print(image)
crop_image = image[top:bottom, left:right]
print(crop_image.shape)
imshow(crop_image)

max_face_mapping = {}
length = len(dataset.all_videos)

scale_down = 4
every_n_frames = 20

assert type(scale_down) is int
assert scale_down >= 1
rescale = 1 / scale_down

print(f'rescale is {rescale}')
json_filename = f'face_map_stats-{scale_down}-{every_n_frames}.json'
print(f'JSON filename is {json_filename}')

for k in tqdm(range(length)):
    filename = dataset.all_videos[k]
    # print(k, filename)
    try:
        vid_obj = dataset.read_video(
            filename, rescale=rescale,
            every_n_frames=every_n_frames
        )
    except ValueError as e:
        print('FAILED TO READ', k, filename)
        continue

    np_frames = vid_obj.out_video
    num_frames = len(np_frames)
    detections = 0
    max_faces = 0

    for i in range(len(np_frames)):
        image = np_frames[i]
        face_locations = face_recognition.face_locations(image)
        faces = len(face_locations)
        max_faces = max(max_faces, faces)

        if faces > 0:
            detections += 1

    p_detection = detections / num_frames
    max_face_mapping[filename] = (max_faces, p_detection)
    print(f'READ {k}/{length}', filename, max_faces, p_detection)


with open(json_filename, 'w') as fp:
    json.dump(max_face_mapping, fp, indent=4)