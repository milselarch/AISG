from RealDataset import RealDataset
from torch.utils import data as data_utils

num_workers = 4
base_location = '../datasets/extract/mtcnn-wav2lip'

if __name__ == '__main__':
    dataset = RealDataset(file_map=base_location, log_on_load=True)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=None, num_workers=num_workers
    )

    k = 0

    for sample in data_loader:
        torch_imgs, torch_mels = sample
        print(f'SAMPLE [{k}] {torch_imgs.shape} {torch_mels.shape}')
        k += 1