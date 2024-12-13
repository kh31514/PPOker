from stable_baselines3.common.callbacks import BaseCallback


class DecayClippingCallback(BaseCallback):
    """Callback to dynamically adjust the clipping value during training."""

    def __init__(self, initial_clip=0.2, clip_decay=0.99, verbose=0):
        super().__init__(verbose)
        self.clip_value = initial_clip
        self.clip_decay = clip_decay

    def _on_step(self) -> bool:
        # Dynamically reduce the clipping value over time
        self.clip_value *= self.clip_decay
        # Log the value for monitoring
        self.logger.record("clip_value", self.clip_value)
        return True
