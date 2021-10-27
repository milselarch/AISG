import datasets
import time
import os

from tqdm.auto import tqdm
from DeepfakeDetection.FaceExtractor import FaceExtractor
from ParallelFaceExtract import ParallelFaceExtract
from PIL import Image

dataset = datasets.Dataset(basedir='datasets')
filenames = dataset.all_videos[:].tolist()
# filenames.append('0ae1576c58393c78.mp4')  # two faces here
# filenames.append('bb34433231a222e5.mp4')  # black background
# filenames.append('0c0c3a74ba96c692.mp4')

filepaths = []
for k in range(len(filenames)):
    filename = filenames[k]
    filepath = f'datasets/train/videos/{filename}'
    filepaths.append(filepath)

start_time = time.perf_counter()
# input(f'IN FILEPATHS {filepaths}')
extractor = ParallelFaceExtract(filepaths=filepaths)
extractor.start(filepaths, num_processes=4)

for k in tqdm(range(len(filenames))):
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
        for i, frame_no in enumerate(faces):
            face = faces[frame_no]
            frame = face.image

            im = Image.fromarray(frame)
            path = f'{face_dir}/{face_no}-{frame_no}.jpg'
            im.save(path)

end_time = time.perf_counter()
duration = end_time - start_time
print(f'extract duration: {duration}')
