from FaceTrainer import FaceTrainer

# trainer = MesoTrainer()
trainer = FaceTrainer(model_type='m', use_cuda=True)

# trainer.train(episodes=4*1000*1000)
# preload_path = 'saves/checkpoints/211211-1551/E85952_T0.59_V0.58.pt'
preload_path = 'saves/checkpoints/211211-2201/E4686912_T0.89_V0.88.pt'
opt_path = 'saves/checkpoints/211211-2201/E4690528-OPT.pt'

trainer.load_optimizer(opt_path)
trainer.load_model(preload_path, eval_mode=False)
trainer.train(episodes=12*1000*1000)