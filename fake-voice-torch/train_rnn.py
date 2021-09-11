from trainer_rnn import Trainer

trainer = Trainer()
trainer.train(
    episodes=500 * 1000, batch_size=32
)