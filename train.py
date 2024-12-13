from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import time
from SB3ActionMaskWrapper import SB3ActionMaskWrapper
from eval import eval_action_mask
from opponent_strats.call import call_focused_strategy
from opponent_strats.random import random_strategy
from save_to_csv import save_to_csv
import numpy as np
import copy

opps = {
    "random": random_strategy,
    "call": call_focused_strategy
}

desired_stats = {"win_rate": [], "total_reward": [],
                 "move_count": [], "move_rate": []}


def create_database(opps, desired_stats):
    data = dict()
    for opp in opps:
        data[opp] = copy.deepcopy(desired_stats)
    data["steps"] = []
    return data


data = create_database(opps, desired_stats)


def train(env_fn, save_folder, steps=32768, step_size=2048, seed=0, clip_range=0.2, callback=None, **env_kwargs):
    env = env_fn.env(**env_kwargs)
    env = SB3ActionMaskWrapper(env)
    env.reset(seed=seed)

    def mask_fn(env):
        return env.action_mask()
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, env,
                        verbose=1, clip_range=clip_range)
    model.set_random_seed(seed)

    for i in range(0, steps, step_size):
        model.learn(total_timesteps=step_size, callback=callback)
        model.save(
            f"saved_models/{save_folder}/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

        for opp in opps:
            strat = opps[opp]
            res = eval_action_mask(
                env_fn, strat, save_folder, num_games=1000, render_mode=None, **env_kwargs
            )
            round_rewards, total_rewards, winrate, scores, moves = res
            data[opp]["move_count"] += [moves]
            data[opp]["move_rate"] += [moves / np.sum(moves)]
            data[opp]["win_rate"] += [winrate]
            data[opp]["total_reward"] += [int(total_rewards['player_1'])]

        data["steps"] += [i+step_size]
        print("Finished training step", i)

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()

    save_to_csv(data, save_folder)

    return
