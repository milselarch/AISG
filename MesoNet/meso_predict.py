from MesoTrainer import MesoTrainer

model_path = 'saves/models/211022-0001.pt'

trainer = MesoTrainer()
trainer.load_model(model_path)
trainer.dataset.label_all_frames(
    predict=trainer.predict_file
)