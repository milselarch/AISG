from MesoTrainer import MesoTrainer

model_path = 'saves/checkpoints/211028-0142/E16296960_T0.87_V0.88.pt'

trainer = MesoTrainer(use_inception=True)
trainer.load_model(model_path)
# trainer.dataset.label_all_frames(predict=trainer.predict_file)
trainer.dataset.label_all_videos()