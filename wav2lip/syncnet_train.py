import torch.multiprocessing as mp

from SyncnetTrainer import SyncnetTrainer

if __name__ == '__main__':
    mp.freeze_support()

    trainer = SyncnetTrainer(
        load_dataset=True,
        train_workers=0, test_workers=0
    )
    trainer.train(episodes=6*1000*1000)
