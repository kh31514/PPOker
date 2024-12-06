from eval import eval_action_mask
# from train.no_clip import train_action_mask
from train.clip_decay import train_action_mask
# from train.chip_clip import train_action_mask
from pettingzoo.classic import texas_holdem_no_limit_v6


env_fn = texas_holdem_no_limit_v6

env_kwargs = {}
steps = 8192

# Train a model against itself (takes ~2 minutes on GPU)
train_action_mask(env_fn, steps=steps, seed=0, **env_kwargs)

# Evaluate 2 games against a random agent
round_rewards, total_rewards, winrate, scores = eval_action_mask(
    env_fn, num_games=100, render_mode=None, **env_kwargs
)

# Watch two games (disabled by default)
# eval_action_mask(env_fn, num_games=2, render_mode="human", **env_kwargs)
