import os

from tqdm.auto import tqdm

base_dir = '../datasets-local/mtcnn-wav2lip'

for sub_dir in tqdm(os.listdir(base_dir)):
    dir_path = f'{base_dir}/{sub_dir}'
    # filenames = sorted(list(os.listdir(dir_path)))
    filenames = list(os.listdir(dir_path))

    for filename in filenames:
        name = filename[:filename.index('.')]
        face_no, frame_no = [int(x) for x in name.split('-')]

        if frame_no % 10 in (5, 6, 7, 8, 9):
            path = f'{dir_path}/{filename}'
            os.remove(path)
            # print(path)
