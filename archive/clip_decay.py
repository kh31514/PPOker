from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import time
from train.mask_fn import mask_fn
from train.SB3ActionMaskWrapper import SB3ActionMaskWrapper
from train.eval import eval_action_mask
from opponent_strats.call import call_focused_strategy
from opponent_strats.random import random_strategy
import pandas as pd
import numpy as np


def train_action_mask(env_fn, steps=10_000, seed=0, initial_clip=0.2, clip_decay=0.99, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking and adaptive clipping."""
    from stable_baselines3.common.callbacks import BaseCallback

    class AdaptiveClippingCallback(BaseCallback):
        """Callback to dynamically adjust the clipping value during training."""

        def __init__(self, initial_clip, clip_decay, verbose=0):
            super().__init__(verbose)
            self.clip_value = initial_clip
            self.clip_decay = clip_decay

        def _on_step(self) -> bool:
            # Dynamically reduce the clipping value over time
            self.clip_value *= self.clip_decay
            # Log the value for monitoring
            self.logger.record("clip_value", self.clip_value)
            return True

    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)

    # Define custom policy with adaptive clipping
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        clip_range=lambda _: initial_clip,  # Use lambda to define dynamic clip value
    )
    model.set_random_seed(seed)

    # Create and add the adaptive clipping callback
    adaptive_clip_callback = AdaptiveClippingCallback(initial_clip, clip_decay)

    call_wr = []
    random_wr = []
    steps_count = []
    call_reward_diff = []
    random_reward_diff = []
    step_size = 2048
    call_df = pd.DataFrame(
        columns=['steps', 'fold', 'call/raise', 'half pot', 'full pot', 'all in'])
    random_df = pd.DataFrame(
        columns=['steps', 'fold', 'call/raise', 'half pot', 'full pot', 'all in'])
    for i in range(0, steps, step_size):
        model.learn(total_timesteps=step_size, callback=adaptive_clip_callback)

        model.save(
            f"saved_models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

        res = eval_action_mask(
            env_fn, call_focused_strategy, num_games=1000, render_mode=None, **env_kwargs
        )
        round_rewards, total_rewards, winrate, scores, moves = res
        print(moves)
        print(moves / np.sum(moves))
        new_row = np.concatenate([[i + step_size], moves / np.sum(moves)])
        print(new_row)
        call_df.loc[len(call_df)] = new_row
        call_wr.append(winrate)
        call_reward_diff.append(int(total_rewards['player_1']))

        res = eval_action_mask(
            env_fn, random_strategy, num_games=1000, render_mode=None, **env_kwargs
        )
        round_rewards, total_rewards, winrate, scores, moves = res
        print(moves)
        new_row = np.concatenate([[i + step_size], moves / np.sum(moves)])
        random_df.loc[len(random_df)] = new_row
        random_reward_diff.append(int(total_rewards['player_1']))
        random_wr.append(float(winrate))

        steps_count.append(i+step_size)
        print(i)

    df = pd.DataFrame()
    df['steps'] = steps_count
    df['call_wr'] = call_wr
    df['call_diff'] = call_reward_diff
    df['random_wr'] = random_wr
    df['random_diff'] = random_reward_diff

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()

    return df, random_df, call_df
