import datasets
import time
import os

from tqdm.auto import tqdm
from DeepfakeDetection.FaceExtractor import FaceExtractor
from NeuralFaceExtract import NeuralFaceExtract
from PIL import Image

dataset = datasets.Dataset(basedir='datasets')
# filenames = dataset.all_videos[:].tolist()
# filenames.append('0ae1576c58393c78.mp4')  # two faces here
# filenames.append('bb34433231a222e5.mp4')  # black background
# filenames.append('0c0c3a74ba96c692.mp4')
# f0d0282ba659ba75
# f45a1fd86d66e669
filenames = ['f45a1fd86d66e669.mp4']

def callback(filepath, face_image_map, pbar):
    name = filepath
    if '/' in filepath:
        name = name[name.rindex('/')+1:]

    name = name[:name.index('.')]
    print(f'NAME {name}')

    export_dir = 'datasets-local/mtcnn-faces-test'
    face_dir = f'{export_dir}/{name}'

    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    if not os.path.exists(face_dir):
        os.mkdir(face_dir)

    # print(f'FACE IMAGE MAP {face_image_map}')

    for face_no in face_image_map:
        faces = face_image_map[face_no]
        for i, frame_no in enumerate(faces):
            face = faces[frame_no]
            frame = face.image

            im = Image.fromarray(frame)
            path = f'{face_dir}/{face_no}-{frame_no}.jpg'
            im.save(path)


filepaths = []

for k in range(len(filenames)):
    filename = filenames[k]
    filepath = f'datasets/train/videos/{filename}'
    filepaths.append(filepath)

start_time = time.perf_counter()
# input(f'IN FILEPATHS {filepaths}')
extractor = NeuralFaceExtract()
extractor.process_filepaths(
    filepaths, every_n_frames=20, batch_size=16,
    callback=callback
)

end_time = time.perf_counter()
duration = end_time - start_time
print(f'extract duration: {duration}')

