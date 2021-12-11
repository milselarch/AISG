from FaceTrainer import FaceTrainer

# trainer = MesoTrainer()
trainer = FaceTrainer(model_type='m')

# trainer.train(episodes=4*1000*1000)
trainer.train(episodes=12*1000*1000)