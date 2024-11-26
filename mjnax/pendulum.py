from typing import Any, Dict, Tuple, TypeVar, Union, Optional
import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from mujoco import mjx
import mujoco

from gymnax.environments.environment import Environment
from gymnax.environments import spaces


MjStateType = TypeVar("MjModelType", bound=mujoco.mjx._src.types.Data)
MjModelType = TypeVar("MjModelType", bound=mujoco.mjx._src.types.Model)


class Pendulum(Environment[MjStateType, MjModelType]):
    """
    Pendulum Environment.

    A Gymnax-integrated environment modeling a single inverted pendulum. 

    Default Attributes:
        xml_path (str):
            Path to the MuJoCo XML model file defining the pendulum's physical properties and configuration.
        n_frame (int):
            Number of frames to maintain in the state buffer for temporal dependencies.
        reward_temp (int):
            Temperature parameter used to scale the reward function, influencing the sensitivity to angle deviations.
        reward_angle (float):
            Threshold angle (in radians) used in the reward calculation to determine the pendulum's deviation from the upright position.
    """
    xml_path: str = "./mjnax/assets/pendulum.xml"
    n_frame: int = 4
    reward_temp: int = 100
    reward_angle: float = jnp.pi / 35
    max_run_time: float = 10.0  # seconds

    @property
    def default_params(self) -> MjModelType:
        return mjx.put_model(self.mj_model)

    @property
    def mj_model(self) -> MjModelType:
        return mujoco.MjModel.from_xml_path(self.xml_path)


    # @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MjStateType,
        action: Union[int, float, chex.Array],
        params: Optional[MjStateType] = None,
    ) -> Tuple[chex.Array, MjStateType, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step without resetting."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        return obs_st, state_st, reward, done, info

    def step_env(self,
                 key: chex.PRNGKey,
                 state: MjStateType,
                 action: Union[int, float, chex.Array],
                 params: MjModelType
                 ) -> Tuple[chex.Array, MjStateType, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Environment-specific step transition."""

        action = self.put_act(action, state, params)

        def _step(index, n_state):
            prev_index = (index + self.n_frame - 1) % self.n_frame
            prev_state = jax.tree_map(lambda s: s[prev_index], n_state)
            prev_state = prev_state.replace(ctrl=jnp.ones_like(prev_state.ctrl) * action)
            next_state = mjx.step(params, prev_state)
            return jax.tree_map(lambda n_s, ns: n_s.at[index].set(ns), n_state, next_state)

        state = jax.lax.fori_loop(0, self.n_frame, _step, state)
        return (self.get_obs(state),
                state,
                self.reward(state, params),
                self.is_terminal(state, params),
                {})

    def reward(self, n_state: MjStateType, params: MjModelType) -> chex.Array:
        pos = n_state.qpos[-1][0] / jnp.pi
        return (- jnp.tanh((pos + 1 - self.reward_angle) * self.reward_temp)
                + jnp.tanh((pos - 1 + self.reward_angle) * self.reward_temp)
                ) / 2 + 1

    def reset_env(self,
                  key: chex.PRNGKey,
                  params: MjModelType
                  ) -> Tuple[chex.Array, MjStateType]:
        """Environment-specific reset."""
        n_state_reset_fn = jax.vmap(lambda _, p: mjx.make_data(p), in_axes=(0, None))
        state = n_state_reset_fn(jrd.split(key, self.n_frame), params)
        return self.get_obs(state), state

    def get_obs(self,
                state: MjStateType,
                ) -> chex.Array:
        """Applies observation function to state."""
        angular_pos = jnp.mod(jnp.abs(state.qpos[-1][0]), jnp.pi) * jnp.sign(state.qpos[-1][0])
        angular_vel = state.qvel[-1][0] / (jnp.pi)
        return jnp.array([angular_pos / jnp.pi, angular_vel])

    def put_act(self,
                act: Union[int, float, chex.Array],
                state: MjStateType,
                params: MjModelType
                ) -> Union[int, float, chex.Array]:
        return act

    def is_terminal(self, state: MjStateType, params: MjModelType) -> jnp.ndarray:
        """Check whether state transition is terminal."""
        # return state.time[-1] >= self.max_run_time
        return False

    def discount(self, state: MjStateType, params: MjModelType) -> jnp.ndarray:
        """Always return a discount of one since environment is only terminated by timeout"""
        return 1.0

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def observation_space(self, params: MjModelType):
        """Observation space of the environment."""
        low = jnp.array(
            [-1, -1],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [1, 1],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (2,), jnp.float32)

    def action_space(self, params: MjModelType):
        """Action space of the environment."""
        low = jnp.array(
            [-1.0],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [1.0],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (2,), jnp.float32)

    def state_space(self, params: MjModelType):
        """State space of the environment."""
        raise NotImplementedError


class DiscretizedPendulum(Pendulum):
    """
    Discretized inverted pendulum environment.

    Extends the Pendulum environment by applying one-hot discretization
    to action and state spaces, allowing finite MDP analysis.
    """

    n_bins = 11

    def get_obs(self,
                state: MjStateType,
                ) -> chex.Array:
        """Applies observation function to state."""
        cont_obs = super().get_obs(state)
        indices = jnp.digitize(cont_obs, jnp.linspace(-1, 1, self.n_bins - 1))
        disc_obs = (indices * (self.n_bins ** jnp.arange(2))).sum()
        return jax.nn.one_hot(disc_obs, self.num_states)

    def put_act(self,
                act: chex.Array,
                state: MjStateType,
                params: MjModelType
                ) -> Union[int, float, chex.Array]:
        return jnp.einsum("a,a->", jnp.linspace(-1, 1, self.num_actions), act)

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 5

    @property
    def num_states(self) -> int:
        """Number of states possible in environment."""
        return self.n_bins ** 2

    def observation_space(self, params: MjModelType):
        """Observation space of the environment."""
        low = jnp.zeros(
            self.num_states,
            dtype=jnp.float32,
        )
        high = jnp.ones(
            self.num_states,
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (self.num_states,), jnp.float32)

    def action_space(self, params: MjModelType):
        """Action space of the environment."""
        low = jnp.zeros(
            self.num_states,
            dtype=jnp.float32,
        )
        high = jnp.ones(
            self.num_states,
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (self.num_actions,), jnp.float32)
