from observe_callback import ObservationBasedClippingCallback
from decay_callback import DecayClippingCallback
from train import train
from pettingzoo.classic import texas_holdem_no_limit_v6

env_fn = texas_holdem_no_limit_v6

steps = 8192
step_size = 2048

train(env_fn, "default", steps=steps, step_size=step_size)

train(env_fn, "no_clip", clip_range=0, steps=steps, step_size=step_size)

decay_callback = DecayClippingCallback()
train(env_fn, "decay_clip", callback=decay_callback, steps=steps, step_size=step_size)

observe_callback = ObservationBasedClippingCallback()
train(env_fn, "observe_clip", callback=observe_callback, steps=steps, step_size=step_size)
