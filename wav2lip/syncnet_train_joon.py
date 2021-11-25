from SyncnetTrainer import SyncnetTrainer

if __name__ == '__main__':
    trainer = SyncnetTrainer(
        train_workers=0, test_workers=0,
        load_dataset=False, pred_ratio=1.0,
        use_joon=True, old_joon=False, dropout_p=0.5,

        face_base_dir='../datasets/extract/mtcnn-sync',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',
        mel_cache_path='saves/preprocessed/mel_cache.npy'
    )

    trainer.start_dataset_workers()
    trainer.ptrain(episodes=6*1000*1000)
