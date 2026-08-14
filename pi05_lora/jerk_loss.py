"""Pi0.5 training model with an optional action-chunk jerk penalty.

This stays in the competition repo and uses Pi0Config.create() as the
extension point, so the vendored openpi checkout does not need a live patch.
"""

from __future__ import annotations

import dataclasses
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

# Real LIBERO actions are [dx, dy, dz, droll, dpitch, dyaw, gripper] (7 dims,
# README.md:99); the model's action_dim=32 slot pads the rest. Only the first
# 6 (continuous end-effector motion) channels belong in a smoothness penalty --
# index 6 (gripper) is a discrete open/close signal, and a grasp/release
# transition would otherwise register as a large "jerk" and bias training
# toward avoiding/delaying gripper actions.
_MOTION_CHANNELS = 6


class JerkPi0(_pi0.Pi0):
    """Pi0 with a training-only second-difference penalty on reconstructed actions."""

    def __init__(self, config: "JerkPi0Config", rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.jerk_loss_weight = config.jerk_loss_weight
        self.jerk_debug = config.jerk_debug

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # Preserve the stock path exactly when disabled: this avoids changing
        # random use, computation, or gradients for all existing runs.
        if self.jerk_loss_weight == 0.0:
            return super().compute_loss(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        base_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        # x_t - time*v_t reconstructs the clean action implied by the model's
        # current flow estimate. Reconstructions at high time are noisier, so
        # (1-time) down-weights their jerk signal; this heuristic is easy to
        # revisit because the whole auxiliary term is controlled by one weight.
        actions_hat = x_t - time_expanded * v_t
        motion = actions_hat[..., :_MOTION_CHANNELS]
        d2 = motion[..., 2:, :] - 2 * motion[..., 1:-1, :] + motion[..., :-2, :]
        jerk_penalty = jnp.mean(jnp.square(d2), axis=(-2, -1)) * (1.0 - time)
        if self.jerk_debug:
            jax.debug.print(
                "jerk_loss_metrics base={base:.6f} jerk_penalty={jerk:.6f} weighted_jerk={weighted:.6f}",
                base=jnp.mean(base_loss),
                jerk=jnp.mean(jerk_penalty),
                weighted=self.jerk_loss_weight * jnp.mean(jerk_penalty),
            )
        return base_loss + self.jerk_loss_weight * jerk_penalty[..., None]


@dataclasses.dataclass(frozen=True)
class JerkPi0Config(pi0_config.Pi0Config):
    """Pi0Config carrying the training-only jerk-loss hyperparameter."""

    jerk_loss_weight: float = 0.0
    # Off by default: printing every compiled loss evaluation ignores the
    # trainer's log_interval and floods stdout with duplicate metric lines on
    # long runs. Opt in explicitly for short debugging runs.
    jerk_debug: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.jerk_loss_weight) or self.jerk_loss_weight < 0:
            raise ValueError(f"jerk_loss_weight must be a finite, non-negative number, got {self.jerk_loss_weight}")

    @override
    def create(self, rng: at.KeyArrayLike) -> JerkPi0:
        return JerkPi0(self, rngs=nnx.Rngs(rng))
