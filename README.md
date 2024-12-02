# MuJoCo Gymnax (Mjnax)

MuJoCo based control environments compatible with [gymnnax](https://github.com/RobertTLange/gymnax).

## How to use

**Interactively render a scene**
```bash
python -m mujoco.viewer --mjcf=./mjnax/assets/cartpole.xml
```

**Passive rendering**

This is only possible in Linux and in mac OS (with ```mjpython```). See [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/python.html) for details.


**Video rendering**

Make a video of a recorded episode.

```Python
import os
import jax
import jax.random as jrd
import jax.numpy as jnp
import mediapy as media

from mjnax.pendulum import Pendulum
from mjnax.render import render_images


env = Pendulum()
params = env.default_params

# Run an episode with random control actions
key, reset_key = jrd.split(jrd.PRNGKey(42), 2)
obs, state = env.reset(key, params)
states = [state]
for _ in range(100):
    key, step_key, act_key = jrd.split(key, 3)
    act = jrd.uniform(act_key, 1, minval=-1, maxval=1)
    next_obs, state, *_ = env.step(step_key, state, act, params)
    states.append(state)
state_seq = jax.tree_map(lambda *s: jnp.concatenate(s), *states)

# Render images and make a video
images = render_images(env, state_seq)
os.makedirs("./videos", exist_ok=True)
delta_step = params.opt.timestep.item()
media.write_video("./videos/pendulum.mp4", images, fps=int(1 / delta_step), qp=1, codec="h264")
```

## Environments

Currently supported environments.

|Name                |Observation Space |Action Space      |
|:------------------:|:----------------:|:----------------:|
|  Pendulum          |Continuous 3      |Continuous 1      |
|  DiscretePendulum  |Discrete 121      |Discrete 3        |
|  Cartpole          |:x:               |:x:               |
|  Swimmer           |:x:               |:x:               |
|  Pusher            |:x:               |:x:               |
|  Reacher           |:x:               |:x:               |

## Benchmarks

:warning: On going ...

## Installation

Recommended: Python 3.11+

Clone the repository and install the package in development mode via:

```bash
pip install -r requirements.txt
pip install -e .
```
