from observe_callback import ObservationBasedClippingCallback
from decay_callback import DecayClippingCallback
from train import train
from pettingzoo.classic import texas_holdem_no_limit_v6

env_fn = texas_holdem_no_limit_v6

train(env_fn, "default")

train(env_fn, "no_clip", clip_range=0)

decay_callback = DecayClippingCallback()
train(env_fn, "decay_clip", callback=decay_callback)

observe_callback = ObservationBasedClippingCallback()
train(env_fn, "observe_clip", callback=observe_callback)
