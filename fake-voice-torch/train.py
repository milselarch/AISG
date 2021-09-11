from trainer import Trainer

trainer = Trainer()
trainer.train(
    episodes=2000 * 1000, batch_size=32
)