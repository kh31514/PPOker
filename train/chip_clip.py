from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import time
from train.mask_fn import mask_fn
from train.SB3ActionMaskWrapper import SB3ActionMaskWrapper


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train with dynamic clipping based on observations."""
    from stable_baselines3.common.callbacks import BaseCallback
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = SB3ActionMaskWrapper(env)
    env.reset(seed=seed)

    env = ActionMasker(env, mask_fn)

    # Define a function for dynamic clip range
    def dynamic_clip_range(progress, observation=None):
        """Dynamic clip range based on training progress and observations."""
        # Example logic: Use the observation to adjust the clip range
        if observation is not None:
            # ind 52 = Number of Chips of player_0 [0, 100]
            # ind 53 = Number of Chips of player_1 [0, 100]
            if (env.agent_selection == "player_0"):
                return observation[52]/100 + 0.1
            else:
                return observation[53]/100 + 0.1
            # return 0.2
            # return dynamic_clip_range(observation)
        # Default fallback
        return 0.2  # or any baseline clip range

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        clip_range=dynamic_clip_range,  # Pass dynamic function
    )
    model.set_random_seed(seed)

    # Example: Add callback to adjust the clipping range based on observations
    class ObservationBasedClippingCallback(BaseCallback):
        """Custom callback to update clipping based on observations."""

        def __init__(self, env, verbose=0):
            super().__init__(verbose)
            self.env = env
            self.current_obs = None  # To hold the current observation

        def _on_step(self) -> bool:
            # Retrieve the latest observation directly from the environment
            # Get observation for the current agent
            self.current_obs = self.env.observe(self.env.agent_selection)
            # Use the current observation to adjust the clipping range
            clip_value = dynamic_clip_range(
                progress=self.model.num_timesteps /
                self.locals["total_timesteps"],
                observation=self.current_obs
            )
            self.logger.record("clip_range", clip_value)  # Log the clip range
            return True

    callback = ObservationBasedClippingCallback(env)
    model.learn(total_timesteps=steps, callback=callback)

    model.save(
        f"saved_models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()
