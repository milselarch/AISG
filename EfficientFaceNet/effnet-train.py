from FaceTrainer import FaceTrainer

# trainer = MesoTrainer()
trainer = FaceTrainer(model_type='m')

# trainer.train(episodes=4*1000*1000)
preload_path = 'saves/checkpoints/211211-1551/E85952_T0.59_V0.58.pt'
trainer.load_model(preload_path, eval_mode=False)
trainer.train(episodes=12*1000*1000)