import time

from SyncDataset import Dataset

dataset = Dataset()

while True:
    dataset.build_samples()
    print('SAMPLES', dataset.get_sample_sizes())
    time.sleep(1)
