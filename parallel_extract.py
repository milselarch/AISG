import datasets
import time
import os

from tqdm.auto import tqdm
from DeepfakeDetection.FaceExtractor import FaceExtractor
from ParallelFaceExtract import ParallelFaceExtract
from PIL import Image

dataset = datasets.Dataset(basedir='datasets')
length = 5

filepaths = []
for k in range(length):
    filename = dataset.all_videos[k]
    filepath = f'datasets/train/videos/{filename}'
    filepaths.append(filepath)

# input(f'IN FILEPATHS {filepaths}')
extractor = ParallelFaceExtract(filepaths=filepaths)
extractor.start(filepaths)

for k in tqdm(range(length)):
    extractions = extractor.extractions
    while len(extractions) == 0:
        print(extractions)
        time.sleep(1)

    keys = list(extractions.keys())
    print(f'EX {keys}')
    export_dir = 'datasets-local/parallel-faces'

    filepath = keys[0]
    face_image_map = extractions[filepath]
    del extractions[filepath]

    name = filepath
    if '/' in filepath:
        name = name[name.rindex('/')+1:]

    name = name[:name.index('.')]
    print(f'NAME {name}')
    face_dir = f'{export_dir}/{name}'
    if not os.path.exists(face_dir):
        os.mkdir(face_dir)

    for face_no in face_image_map:
        faces = face_image_map[face_no]
        for i, frame in enumerate(faces):
            im = Image.fromarray(frame)
            path = f'{face_dir}/{face_no}-{i}.jpg'
            im.save(path)