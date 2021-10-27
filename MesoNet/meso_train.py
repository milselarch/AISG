from MesoTrainer import MesoTrainer

# trainer = MesoTrainer()
trainer = MesoTrainer(use_inception=True)

# trainer.train(episodes=4*1000*1000)
trainer.train(episodes=12*1000*1000)