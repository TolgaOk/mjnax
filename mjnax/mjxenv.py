from typing import Any, Dict, Tuple, TypeVar, Union, Optional
from functools import partial
import os
import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from mujoco import mjx
import mujoco
from gymnax.environments.environment import Environment

import mjnax


MjxStateType = TypeVar("MjxModelType", bound=mujoco.mjx._src.types.Data)
MjxModelType = TypeVar("MjxModelType", bound=mujoco.mjx._src.types.Model)


class MjxEnvironment(Environment[MjxStateType, MjxModelType]):
    """ Base Mjx environment class.
    """
    xml_path: str
    n_repeat_act: int

    @property
    def default_params(self) -> MjxModelType:
        return mjx.put_model(self.mj_model)

    @property
    def mj_model(self) -> MjxModelType:
        absolute_path = os.path.join(mjnax.__path__._path[0], self.xml_path)
        return mujoco.MjModel.from_xml_path(absolute_path)

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: chex.PRNGKey,
             state: MjxStateType,
             action: Union[int, float, chex.Array],
             params: Optional[MjxStateType] = None,
             ) -> Tuple[chex.Array,
                        MjxStateType,
                        jnp.ndarray,
                        jnp.ndarray,
                        Dict[Any, Any]]:
        """ Perform one step in MuJoCo.
            Auto resetting in termination is omitted.
            The returned state reflects the application of the given action repeated
            self.n_repeat_act times, encapsulating all intermediate steps in the resulting 
            mjx_state.

        Args:
            key (chex.PRNGKey): rng
            state (MjxStateType): vectorized mjx state
            action (Union[int, float, chex.Array]): control action
            params (Optional[MjxStateType], optional): mjx model. Defaults to None.

        Returns:
            Tuple[chex.Array, MjxStateType, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
            obs, state, reward, terminal, info
        """
        if params is None:
            params = self.default_params

        step_key, reward_key, terminal_key, obs_key = jrd.split(key, 4)

        next_state = self.step_mjx(step_key, state, action, params)
        return (self.get_obs(obs_key, next_state, params),
                next_state,
                self.calculate_reward(reward_key, state, next_state, action, params),
                self.is_terminal(terminal_key, next_state, params),
                {})

    def step_mjx(self,
                 key: chex.PRNGKey,
                 state: MjxStateType,
                 action: Union[int, float, chex.Array],
                 params: MjxStateType
                 ) -> MjxStateType:
        """ Step MuJoCo n times with the same control action.

        Args:
            key (chex.PRNGKey): rng
            state (MjxStateType): vectorized mjx state
            action (Union[int, float, chex.Array]): control action
            params (MjxStateType): mjx model

        Returns:
            MjxStateType: next vectorized mjx state after n MuJoCo steps
        """
        def mjx_step(index: int, state: MjxStateType) -> MjxStateType:
            prev_index = (index + self.n_repeat_act - 1) % self.n_repeat_act
            prev_state = jax.tree_map(lambda s: s[prev_index], state)
            prev_state = prev_state.replace(ctrl=jnp.ones_like(prev_state.ctrl) * action)
            next_state = mjx.step(params, prev_state)
            return jax.tree_map(lambda n_s, ns: n_s.at[index].set(ns), state, next_state)

        return jax.lax.fori_loop(0, self.n_repeat_act, mjx_step, state)

    def reset_env(self,
                  key: chex.PRNGKey,
                  params: MjxModelType
                  ) -> Tuple[chex.Array, MjxStateType]:
        """ Reset the mjx environment.
            The returned mjx_state is repeated self.n_repeat_act times.

        Args:
            key (chex.PRNGKey): rng
            params (MjxModelType): mjx model

        Returns:
            Tuple[chex.Array, MjxStateType]:
            obs, mjx model
        """
        reset_key, obs_key = jrd.split(key, 2)
        n_state_reset_fn = jax.vmap(lambda _, p: mjx.make_data(p), in_axes=(0, None))
        state = n_state_reset_fn(jrd.split(reset_key, self.n_repeat_act), params)
        return self.get_obs(obs_key, state, params), state

    def is_terminal(self,
                    key: chex.PRNGKey,
                    state: MjxStateType,
                    params: MjxModelType
                    ) -> chex.Array:
        """ Check whether the state is terminal.

        Args:
            state (MjxStateType): vectorized mjx state
            params (MjxModelType): mjx model

        Returns:
            chex.Array: Boolean value indicating the termination
        """
        raise NotImplementedError

    def calculate_reward(self,
                         key: chex.PRNGKey,
                         state: MjxStateType,
                         next_state: MjxStateType,
                         action: Union[int, float, chex.Array],
                         params: MjxModelType
                         ) -> chex.Array:
        """ Calculate reward of the state transition

        Args:
            state (MjxStateType): vectorized mjx state before transition
            next_state (MjxStateType): vectorized mjx state after transition
            action (Union[int, float, chex.Array]): control action
            params (MjxModelType): mjx model

        Returns:
            chex.Array: Reward of the transition
        """
        raise NotImplementedError

    def get_obs(self,
                key: chex.PRNGKey,
                state: MjxStateType,
                params: MjxModelType
                ) -> chex.Array:
        """ Observation of the mjx state.

        Args:
            key (chex.PRNGKey): rng
            state (MjxStateType): vectorized mjx state
            params (MjxModelType): mjx model

        Returns:
            chex.Array: Observation
        """
        raise NotImplementedError
