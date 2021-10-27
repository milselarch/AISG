import pandas as pd

from NeuralFaceExtract import NeuralFaceExtract

mismatch_path = 'stats/mismatches-211027-1914.csv'
mismatch_df = pd.read_csv(mismatch_path)
mismatch_filenames = mismatch_df['filename'].to_numpy()

extractor = NeuralFaceExtract()
extractor.extract_all(
    mismatch_filenames, every_n_frames=20
)