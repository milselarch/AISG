from RealDataset import RealDataset
from torch.utils import data as data_utils

num_workers = 0
base_location = '../datasets/extract/mtcnn-wav2lip'
dataset = RealDataset(file_map=base_location)
data_loader = data_utils.DataLoader(
    dataset, batch_size=None, num_workers=num_workers
)

for sample in dataset:
    torch_imgs, torch_mels = sample
    print(f'SAMPLE {torch_imgs.shape} {torch_mels.shape}')