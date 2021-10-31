import pandas as pd
import datasets

from NeuralFaceExtract import NeuralFaceExtract

dataset = datasets.Dataset(basedir='datasets')
filenames = dataset.all_videos[:].tolist()

extractor = NeuralFaceExtract()
extractor.export_dir = 'datasets-local/mtcnn-wav2lip'
extractor.extract_all(
    filenames, every_n_frames=1,
    skip_detect=10, export_size=96
)