from typing import List
import numpy as np
import mujoco
from mjnax.mjxenv import MjxEnvironment, MjxStateType


def render_images(env: MjxEnvironment,
                  state_seq: MjxStateType,
                  height: int = 1000,
                  width: int = 1500,
                  camera: int = -1
                  ) -> List[np.array]:
    """ Render mjx environment by applying the same control actions in MuJoCo.

    Args:
        env (MjxEnvironment): mjx environment
        state_seq (MjxStateType): vectorized mjx state containing the states of an episode.
        height (int, optional): rendering height in px. Defaults to 1000.
        width (int, optional): rendering width in px. Defaults to 1500.
        camera (int, optional): camera index. Defaults to -1 (default camera index).

    Returns:
        List[np.array]: Rendered images array
    """
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    n_steps = len(state_seq.ctrl)

    renderer.update_scene(mj_data)
    images = [renderer.render()]

    for index in range(n_steps):
        mj_data.ctrl = np.array(state_seq.ctrl[index])
        mujoco.mj_step(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera)
        images.append(renderer.render())

    return images
