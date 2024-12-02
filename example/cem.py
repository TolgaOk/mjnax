""" Implementation of Cross Entropy Method [1] in Pendulum
    [1] Rubinstein, R. Y., & Kroese, D. P. (2004). A Tutorial Introduction to the Cross-Entropy
    Method. In The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization,
    Monte-Carlo Simulation and Machine Learning (pp. 29–58). Springer New York.
    https://doi.org/10.1007/978-1-4757-4321-0_2
"""
from typing import Callable, Any, Tuple
from functools import partial
import os
import jax
import jax.random as jrd
import jax.numpy as jnp
from flax import struct
import chex
import mediapy as media

from mjnax.pendulum import Pendulum
from mjnax.render import render_images
from mjnax.mjxenv import MjxEnvironment, MjxStateType, MjxModelType

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["policy", "env"])
def run_episode(key: chex.PRNGKey,
                env: MjxEnvironment,
                env_param: MjxModelType,
                policy_param: Any,
                policy: Callable,
                episode_len: int = 100
                ) -> Tuple[chex.Array, MjxStateType]:
    """ Run one episode by following the policy.

    Args:
        key (chex.PRNGKey): rng
        env (MjxEnvironment): mjx environment
        policy (Callable): deterministic policy function

    Returns:
        Tuple[chex.Array, MjxStateType]
        reward array and mjx state squence
    """
    key, reset_key = jrd.split(key, 2)
    obs, state = env.reset(reset_key, env_param)
    rewards = jnp.full(episode_len, jnp.nan)

    state_seq = jax.tree.map(lambda x: jnp.full(((episode_len + 1), *x.shape), jnp.nan), state)
    state_seq = jax.tree.map(lambda x, s: x.at[0].set(s), state_seq, state)

    def _step(index, payload):
        key, env_state, rewards, obs, state_seq = payload
        key, step_key, act_key = jrd.split(key, 3)
        act = policy(act_key, obs, policy_param)
        obs, env_state, reward, terminal, *_ = env.step(step_key, env_state, act, env_param)
        rewards = rewards.at[index].set(reward)
        state_seq = jax.tree.map(lambda x, s: x.at[index + 1].set(s), state_seq, env_state)
        return key, env_state, rewards, obs, state_seq

    key, env_state, rewards, obs, state_seq = jax.lax.fori_loop(
        0, episode_len, _step, (key, state, rewards, obs, state_seq))

    return rewards, state_seq


@struct.dataclass
class PolicyParam:
    W_1: chex.Array
    b_1: chex.Array
    W_2: chex.Array
    b_2: chex.Array


def policy(key: chex.PRNGKey,
           obs: chex.Array,
           policy_param: PolicyParam
           ) -> chex.Array:
    """One hidden layer neural network function.

    Args:
        key (chex.PRNGKey): rng 
        obs (chex.Array): Observation from the environment.
        policy_param (PolicyParam): Parameters of the policy.

    Returns:
        Action
    """
    logit = jnp.dot(obs, policy_param.W_1) + policy_param.b_1
    action = jnp.clip(jnp.dot(jax.nn.relu(logit), policy_param.W_2) + policy_param.b_2, -1, 1)
    return action


def cem_step(key: chex.PRNGKey,
             env: Pendulum,
             env_params: MjxModelType,
             mean: PolicyParam,
             cov: PolicyParam,
             sample_size: int,
             elite_frac: float,
             min_cov: float
             ) -> Tuple[PolicyParam, PolicyParam, float]:
    """Performs one iteration of the Cross-Entropy Method.

    Args:
        key: PRNG key.
        env: The environment.
        env_params: Parameters for the environment.
        mean: Mean of the policy parameter distribution.
        cov: Covariance of the policy parameter distribution.
        sample_size: Number of samples to generate.
        elite_frac: Fraction of top-performing samples to consider elite.
        min_cov: Minimum covariance

    Returns:
        Updated mean, covariance, and average reward.
    """
    param_samples = jax.tree.map(lambda m, c: jrd.multivariate_normal(key, m.flatten(), c, (sample_size,)).reshape(sample_size, *m.shape),
                                 mean, cov)
    keys = jrd.split(key, sample_size)

    def eval_sample(params_and_key):
        pi_param, key = params_and_key
        rewards, _ = run_episode(key, env, env_params, pi_param, policy)
        return rewards.sum()

    # Evaluate all samples
    rewards = jax.vmap(eval_sample)((param_samples, keys))

    # Select elite samples
    num_elite = max(int(sample_size * elite_frac), 1)
    elite_indices = jnp.argsort(rewards)[-num_elite:]
    elite_params = jax.tree.map(lambda p: p[elite_indices], param_samples)

    # Update distribution parameters
    new_mean = jax.tree.map(lambda p: jnp.mean(p, axis=0), elite_params)
    new_cov = jax.tree.map(
        lambda p: jnp.cov(p.reshape(num_elite, -1).T)
        + jnp.eye(p.reshape(num_elite, -1).shape[-1]) * min_cov,
        elite_params)

    avg_reward = jnp.mean(rewards)

    return new_mean, new_cov, avg_reward


if __name__ == "__main__":
    env = Pendulum()
    env_params = env.default_params
    key = jrd.PRNGKey(42)

    hidden_size = 10
    sample_size = 50
    elite_frac = 0.3
    num_iterations = 500
    min_cov = 0.25

    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.num_actions

    # Initialize mean and covariance for the policy parameter distribution
    mean = PolicyParam(W_1=jnp.zeros((obs_dim, hidden_size)),
                       b_1=jnp.zeros(hidden_size),
                       W_2=jnp.zeros((hidden_size, action_dim)),
                       b_2=jnp.zeros(action_dim))
    cov = PolicyParam(
        W_1=jnp.eye(obs_dim * hidden_size),
        b_1=jnp.eye(hidden_size),
        W_2=jnp.eye(hidden_size * action_dim),
        b_2=jnp.eye(action_dim),
    )

    for i in range(num_iterations):
        key, subkey = jrd.split(key)
        mean, cov, avg_reward = cem_step(
            subkey, env, env_params, mean, cov, sample_size, elite_frac, min_cov
        )
        print(f"Iteration {i+1}, Average Reward: {avg_reward}")

    # Run an episode with the optimized policy
    key, subkey = jrd.split(key)
    rewards, stacked_steps = run_episode(subkey, env, env_params, mean, policy)
    episode = jax.tree.map(lambda x: x.reshape(
        x.shape[0] * x.shape[1], *x.shape[2:]), stacked_steps)
    print(f"Final Reward: {rewards.sum().item()}")

    images = render_images(env, episode)
    os.makedirs("./videos", exist_ok=True)
    delta_step = env_params.opt.timestep.item()
    media.write_video("./videos/pendulum-cem.mp4", images,
                      fps=int(1 / delta_step), qp=1, codec="h264")
