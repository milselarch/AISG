from trainer import Trainer

trainer = Trainer(
    cache_threshold=20, use_batch_norm=True,
    add_aisg=True, use_avs=False, train_version=1
)

trainer.load_model(
    'saves/checkpoints/211001-1108/E149152_T0.51_V0.51.pt',
    eval_mode=False
)
trainer.train(
    episodes=1000 * 1000, batch_size=32
)

# LA_T_7717952.flac
# 210921-2305.pt