from SyncnetTrainer import SyncnetTrainer

# preload_path = 'saves/checkpoints/211125-1900/E6143040_T0.77_V0.66.pt'
preload_path = 'saves/checkpoints/211202-0328/E10695968_T0.84_V0.69.pt'

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
        use_cuda=False, load_dataset=False,
        use_joon=True, old_joon=False,
        preload_path=preload_path,

        face_base_dir='../datasets/extract/mtcnn-lip',
        video_base_dir='../datasets/train/videos',
        audio_base_dir='../datasets/extract/audios-flac',
        mel_cache_path='saves/preprocessed/mel_cache.npy',

        fcc_list=(512, 128, 32),
        pred_ratio=1.0,  dropout_p=0.5,
        is_checkpoint=False, predict_confidence=True,
        transform_image=True, eval_mode=True
    )

    trainer.load_model(preload_path, eval_mode=True)
    trainer.model.disable_norm_toggle()
    mono_filename = True
    is_training = True

    trainer.start_dataset_workers(
        start_train_workers=is_training,
        start_test_workers=not is_training,
        image_cache_workers=1
    )
    trainer.validate(
        episodes=500, fake_p=0.5, mono_filename=mono_filename,
        use_train_data=is_training
    )

    print('MONO FILENAME', mono_filename)
    print('END')