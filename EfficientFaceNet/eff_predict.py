from FaceTrainer import FaceTrainer

model_path = 'saves/checkpoints/211211-2201/E4686912_T0.89_V0.88.pt'
# face predictions exported to ../stats/face-predictions-211106-0928.csv

trainer = FaceTrainer(
    model_type='m', load_dataset=True,
    use_cuda=True
)

trainer.load_model(model_path, map_location='cpu')
trainer.dataset.label_all_frames(
    predict=trainer.predict_file, max_samples=16, clip=None
)