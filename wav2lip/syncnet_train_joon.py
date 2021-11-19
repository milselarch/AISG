from SyncnetTrainer import SyncnetTrainer

if __name__ == '__main__':
    trainer = SyncnetTrainer(
        train_workers=0, test_workers=0,
        load_dataset=False, pred_ratio=1.0,
        use_joon=True,

        face_base_dir='../datasets/extract/mtcnn-sync',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac'
    )

    trainer.start_dataset_workers()
    trainer.train(episodes=6*1000*1000)
