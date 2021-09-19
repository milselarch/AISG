from wav_trainer import Trainer

trainer = Trainer()
trainer.train(
    episodes=2000 * 10000, batch_size=16
)