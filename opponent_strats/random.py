def random_strategy(agent, env, action_mask):
    return env.action_space(agent).sample(action_mask)
