def call_focused_strategy(agent, env, action_mask):
    # Identify the "call" action's index in the action space
    CALL_ACTION_INDEX = 1  # Replace with the actual index for "call" in your environment

    # Check if "call" is valid
    if action_mask[CALL_ACTION_INDEX]:
        return CALL_ACTION_INDEX

    # Fallback: If "call" isn't valid, choose a random valid action
    return env.action_space(agent).sample(action_mask)
