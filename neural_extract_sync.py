import pandas as pd
import datasets
import datetime
import traceback
import cProfile

from NeuralFaceExtract import NeuralFaceExtract

dt = datetime.datetime.now()
stamp = dt.strftime('%Y%m%d-%H%M%S')
profile_path = f'stats/neural-extract-{stamp}.profile'

dataset = datasets.Dataset(basedir='datasets')
filenames = dataset.all_videos[:].tolist()
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
    extractor.export_dir = 'datasets/extract/mtcnn-sync'

    try:
        extractor.extract_all(
            filenames, every_n_frames=1,
            skip_detect=10, export_size=256,
            ignore_detect=20
        )
    except Exception as e:
        print(traceback.format_exc())
        print('EXTRACTION FAILED')

    profile.disable()
    profile.dump_stats(profile_path)
    print(f'profile save to {profile_path}')