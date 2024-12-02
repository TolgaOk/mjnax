from typing import List
import numpy as np
import jax
import mujoco
from mjnax.mjxenv import MjxEnvironment, MjxStateType


def render_images(env: MjxEnvironment,
                  state_seq: MjxStateType,
                  height: int = 1000,
                  width: int = 1500,
                  camera: int = -1,
                  check_mismatch: bool = False
                  ) -> List[np.array]:
    """ Render mjx environment in MuJoCo renderer.

    Args:
        env (MjxEnvironment): mjx environment
        state_seq (MjxStateType): vectorized mjx state containing the states of an episode.
        height (int, optional): rendering height in px. Defaults to 1000.
        width (int, optional): rendering width in px. Defaults to 1500.
        camera (int, optional): camera index. Defaults to -1 (default camera index).

    Raises:
        RuntimeError: If check_mismatch is true and there is a mismatch in qpos between
        MuJoCo and MJX states.

    Returns:
        List[np.array]: Rendered images array
    """
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    state_seq = jax.tree.map(lambda x: x[env.n_repeat_act - 1:], state_seq)
    n_steps = len(state_seq.ctrl)

    renderer.update_scene(mj_data)
    images = [renderer.render()]

    for index in range(n_steps - 1):
        error = np.allclose(mj_data.qpos, state_seq.qpos[index], atol=1e-5)
        if check_mismatch and error:
            raise RuntimeError(f"Mismatch between MJX and MuJoCo at step: {index}")
        mj_data.ctrl = np.array(state_seq.ctrl[index + 1])
        mujoco.mj_step(mj_model, mj_data)

        mj_data.qpos = np.array(state_seq.qpos[index + 1])
        renderer.update_scene(mj_data, camera=camera)
        images.append(renderer.render())

    return images
