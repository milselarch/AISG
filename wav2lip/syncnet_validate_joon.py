from SyncnetTrainer import SyncnetTrainer

preload_path = 'saves/checkpoints/211125-1900/E6143040_T0.77_V0.66.pt'

"""
validation data: fake_p = 0
accuracy: 0.7646484375
mean error: 0.32524338364601135
mean squared error: 0.45284234471820733
average pred: 0.32524338364601135

validation data: fake_p = 1
accuracy: 0.8583984375
mean error: 0.1470394730567932
mean squared error: 0.3089771303661026
average pred: 0.8529605269432068

validation data: fake_p = 0 [mono]
accuracy: 0.2236328125
mean error: 0.7659800052642822
mean squared error: 0.8149596178546675
average pred: 0.7659800052642822
"""

if __name__ == '__main__':
    trainer = SyncnetTrainer(
        train_workers=0, test_workers=0,
        load_dataset=False, pred_ratio=1.0,
        use_joon=True, old_joon=False, dropout_p=0.5,
        fcc_list=(512, 128, 32),

        face_base_dir='../datasets/extract/mtcnn-sync',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',
        mel_cache_path='saves/preprocessed/mel_cache.npy',
        transform_image=True,
    )

    trainer.load_model(preload_path, eval_mode=True)
    trainer.start_dataset_workers()
    trainer.validate(
        episodes=1000, fake_p=0, mono_filename=True
    )
