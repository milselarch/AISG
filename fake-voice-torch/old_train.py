from old_trainer import Trainer

trainer = Trainer(cache_threshold=20)
trainer.train(
    episodes=1000 * 1000, batch_size=32
)

# LA_T_7717952.flac
