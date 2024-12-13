from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import time
from train.SB3ActionMaskWrapper import SB3ActionMaskWrapper
from train.mask_fn import mask_fn
from train.eval import eval_action_mask
from opponent_strats.call import call_focused_strategy
from opponent_strats.random import random_strategy
import pandas as pd
import numpy as np


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)

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
        model.learn(total_timesteps=step_size)
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
