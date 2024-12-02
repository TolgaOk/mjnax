from tabnanny import check
from typing import Any, Dict, Tuple, TypeVar, Union, Optional
from functools import partial
import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from mujoco import mjx
import mujoco

from gymnax.environments.environment import Environment
from gymnax.environments import spaces
from mjnax.mjxenv import MjxEnvironment, MjxModelType, MjxStateType


class Pendulum(MjxEnvironment):
    """
    Pendulum Environment.

    A Gymnax-integrated environment modeling a single inverted pendulum. 

    Default Attributes:
        xml_path (str):
            Path to the MuJoCo XML model file defining the pendulum's physical properties and configuration.
        n_repeat_act (int):
            Number of frames to maintain in the state buffer for temporal dependencies.
        reward_temp (int):
            Temperature parameter used to scale the reward function, influencing the sensitivity to angle deviations.
        reward_angle (float):
            Threshold angle (in radians) used in the reward calculation to determine the pendulum's deviation from the upright position.
    """
    xml_path: str = "assets/pendulum.xml"
    n_repeat_act: int = 4
    reward_temp: int = 100
    reward_angle: float = jnp.pi / 35

    def calculate_reward(self,
                         key: chex.PRNGKey,
                         state: MjxStateType,
                         next_state: MjxStateType,
                         action: Union[int, float, chex.Array],
                         params: MjxModelType
                         ) -> chex.Array:
        """ Positive reward for upright position. Reward is in range [0, 1]"""
        angular_pos_cos = jnp.cos(state.qpos[:][0])
        return (-angular_pos_cos).min()

    def get_obs(self,
                key: chex.PRNGKey,
                state: MjxStateType,
                params: MjxModelType
                ) -> chex.Array:
        """ Cosine and sine of the angular position and angular velocity """
        angular_pos_sin = jnp.sin(state.qpos[-1][0])
        angular_pos_cos = jnp.cos(state.qpos[-1][0])
        angular_vel = state.qvel[-1][0] / (jnp.pi)
        return jnp.array([angular_pos_sin, angular_pos_cos, angular_vel])

    def is_terminal(self,
                    key: chex.PRNGKey,
                    state: MjxStateType,
                    params: MjxModelType
                    ) -> chex.Array:
        """ Always false (infinite horizon) """
        return False

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def observation_space(self, params: MjxModelType):
        """Observation space of the environment."""
        low = jnp.array(
            [-1, -1, -1],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [1, 1, 1],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (3,), jnp.float32)

    def action_space(self, params: MjxModelType):
        """Action space of the environment."""
        low = jnp.array(
            [-1.0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [1.0],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (1,), jnp.float32)


class DiscretizedPendulum(Pendulum):
    """
    Discretized inverted pendulum environment.

    Extends the Pendulum environment by applying one-hot discretization
    to action and state spaces, allowing finite MDP analysis.
    """

    n_bins = 11

    def get_obs(self,
                key: chex.PRNGKey,
                state: MjxStateType,
                params: MjxModelType
                ) -> chex.Array:
        """Applies observation function to state."""
        pos = state.qpos[-1][0] / jnp.pi
        pos_index = jnp.digitize(
            jnp.mod(pos / 2 - 0.5, 1) * 2 - 1,
            jnp.linspace(-1, 1, self.n_bins)) - 1
        vel = state.qvel[-1][0] / jnp.pi
        vel_index = jnp.digitize(
            jnp.clip(vel, -1, 1),
            jnp.linspace(-1, 1, 11)) - 1
        disc_obs = (jnp.array([pos_index, vel_index]) * (self.n_bins ** jnp.arange(2))
                    ).sum()
        return jax.nn.one_hot(disc_obs, self.num_states)

    def step_mjx(self,
                 key: chex.PRNGKey,
                 action: chex.Array,
                 state: MjxStateType,
                 params: MjxModelType
                 ) -> Union[int, float, chex.Array]:
        """ Step Pendulum by converting the discrete actions to continuous ones. """
        cont_action = jnp.einsum("a,a->", jnp.linspace(-1, 1, self.num_actions), action)
        return super().step_mjx(key, state, cont_action, params)

    @property
    def num_actions(self) -> int:
        """ Number of actions possible in environment. """
        return 3

    @property
    def num_states(self) -> int:
        """ Number of states possible in environment. """
        return self.n_bins ** 2

    def observation_space(self, params: MjxModelType):
        """ One hot observation space of the environment. """
        low = jnp.zeros(
            self.num_states,
            dtype=jnp.float32,
        )
        high = jnp.ones(
            self.num_states,
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (self.num_states,), jnp.float32)

    def action_space(self, params: MjxModelType):
        """ One hot action space of the environment. """
        low = jnp.zeros(
            self.num_states,
            dtype=jnp.float32,
        )
        high = jnp.ones(
            self.num_states,
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (self.num_actions,), jnp.float32)
