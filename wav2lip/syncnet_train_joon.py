from SyncnetTrainer import SyncnetTrainer

if __name__ == '__main__':
    trainer = SyncnetTrainer(
        train_workers=0, test_workers=0,
        load_dataset=False, pred_ratio=1.0,
        use_joon=True, old_joon=False, dropout_p=0.5,
        fcc_list=(512, 128, 32),

        face_base_dir='../datasets/extract/mtcnn-lip',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',
        mel_cache_path='saves/preprocessed/mel_cache.npy',

        transform_image=True, predict_confidence=True,
        binomial_train_sampling=True
    )

    trainer.start_dataset_workers(
        start_train_workers=True, start_test_workers=False,
        image_cache_workers=1
    )
    trainer.ptrain(episodes=20*1000*1000)
