import pandas as pd
import datetime
import traceback
import cProfile

import ParentImport
import datasets

from NeuralFaceExtract import NeuralFaceExtract

dt = datetime.datetime.now()
stamp = dt.strftime('%Y%m%d-%H%M%S')
profile_path = f'profiles/neural-extract-{stamp}.profile'

basedir = '../datasets'
video_base_dir = f'{basedir}/train/videos'

dataset = datasets.Dataset(basedir=basedir)
filenames = dataset.all_videos[:].tolist()
# filenames = ['e39a5b7f32cac303.mp4']
# filenames = ['7f158571c5cdcd1f.mp4']
# filenames = ['9584bf852635aabe.mp4']
# filenames = ['15131c1ca037f1f8.mp4']
# filenames = ['0a4da6b49315507c.mp4']
# filenames = ['a09d90703e755a13.mp4']
# filenames = ['f45a1fd86d66e669.mp4']
# filenames = ['a161b256a9dcd783.mp4']  # empty

if __name__ == '__main__':
    profile = cProfile.Profile()
    profile.enable()

    extractor = NeuralFaceExtract()
    extractor.export_dir = '../datasets/extract/mtcnn-lip'

    try:
        extractor.extract_all(
            filenames, basedir=basedir,
            video_base_dir=video_base_dir,
            every_n_frames=1, skip_detect=5,
            export_size=256, ignore_detect=20,
            save_mouth=True
        )
    except Exception as e:
        print(traceback.format_exc())
        print('EXTRACTION FAILED')

    profile.disable()
    profile.dump_stats(profile_path)
    print(f'profile save to {profile_path}')