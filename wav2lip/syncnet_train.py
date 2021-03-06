import torch.multiprocessing as mp

if __name__ == '__main__':
    from SyncnetTrainer import SyncnetTrainer

    mp.freeze_support()

    trainer = SyncnetTrainer(
        train_workers=0, test_workers=0,
        load_dataset=False
    )

    trainer.start_dataset_workers()
    trainer.train(episodes=6*1000*1000)
