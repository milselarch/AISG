from MesoTrainer import MesoTrainer

model_path = 'saves/models/211022-0001.pt'
# face predictions exported to ../stats/face-predictions-211106-0928.csv

trainer = MesoTrainer()
trainer.load_model(model_path)
trainer.dataset.label_all_frames(
    predict=trainer.predict_file
)