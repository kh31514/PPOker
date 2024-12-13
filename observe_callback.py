

class ObservationBasedClippingCallback(BaseCallback):
    """Custom callback to update clipping based on observations."""

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.current_obs = None  # To hold the current observation

    def dynamic_clip_range(self, progress, observation=None):
        """Dynamic clip range based on training progress and observations."""
        # Example logic: Use the observation to adjust the clip range
        if observation is not None:
            # ind 52 = Number of Chips of player_0 [0, 100]
            # ind 53 = Number of Chips of player_1 [0, 100]
            if (self.env.agent_selection == "player_0"):
                return observation[52]/100 + 0.1
            else:
                return observation[53]/100 + 0.1
        # Default fallback
        return 0.2  # or any baseline clip range

    def _on_step(self) -> bool:
        # Retrieve the latest observation directly from the environment
        # Get observation for the current agent
        self.current_obs = self.env.observe(self.env.agent_selection)
        # Use the current observation to adjust the clipping range
        clip_value = self.dynamic_clip_range(
            progress=self.model.num_timesteps /
            self.locals["total_timesteps"],
            observation=self.current_obs
        )
        self.logger.record("clip_range", clip_value)  # Log the clip range
        return True
