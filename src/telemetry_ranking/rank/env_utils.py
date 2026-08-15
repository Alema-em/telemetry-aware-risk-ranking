"""Helpers for StepCountJITAI counterfactual ranking."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
from StepCountJITAI.envs import StepCountJITAI


def make_env(
    sigma: float = 0.4,
    seed: int = 0,
    n_version: int = 0,
    chosen_obs_names: list[str] | None = None,
    **env_overrides: Any,
) -> StepCountJITAI:
    """Create a ranking-friendly env. n_version=0 = deterministic dynamics.

    Optional overrides support the H/D ablation. Defaults preserve the
    reference environment.
    """
    if chosen_obs_names is None:
        chosen_obs_names = ["P", "H", "D"]
    kwargs: dict[str, Any] = dict(
        sigma=sigma,
        chosen_obs_names=chosen_obs_names,
        n_version=n_version,
        b_display=False,
        seed=seed,
        b_using_uniform=True,
        chosen_a=0.01,
    )
    if n_version == 1:
        kwargs["σs"] = 10.0
    kwargs.update(env_overrides)
    return StepCountJITAI(**kwargs)


def snapshot(env: StepCountJITAI) -> dict[str, Any]:
    return {
        "current_state": np.array(env.current_state, dtype=float).copy(),
        "current_t": int(env.current_t),
        "obs": copy.deepcopy(env.obs),
        "rng0_state": env.rng0.bit_generator.state,
        "rng1_state": env.rng1.bit_generator.state,
    }


def restore(env: StepCountJITAI, snap: dict[str, Any]) -> None:
    env.current_state = np.array(snap["current_state"], dtype=float).copy()
    env.current_t = int(snap["current_t"])
    env.obs = copy.deepcopy(snap["obs"])
    env.rng0.bit_generator.state = snap["rng0_state"]
    env.rng1.bit_generator.state = snap["rng1_state"]


def set_true_context(env: StepCountJITAI, c: int) -> None:
    """Overwrite latent true context used by dynamics (for counterfactual sims)."""
    d = env.unpack_obs_array(env.current_state)
    env.current_state = env.create_obs_array(
        int(c),
        list(d["probs"]),
        int(d["c_infer"]),
        float(d["h"]),
        float(d["d"]),
        int(d["z"]),
        int(d["w"]),
        float(d["s"]),
    )
    env.obs = env.extract_obs(env.chosen_obs_names, env.b_add_time_inhomogeneous)


def belief_probs(env: StepCountJITAI) -> np.ndarray:
    d = env.unpack_obs_array(env.current_state)
    return np.asarray(d["probs"], dtype=float)


def context_features(env: StepCountJITAI) -> np.ndarray:
    """Numeric features available to a ranker (no true C)."""
    d = env.unpack_obs_array(env.current_state)
    # probs for both contexts, inferred L, H, D, S, time indicator
    return np.array(
        [
            float(d["probs"][0]),
            float(d["probs"][1]) if len(d["probs"]) > 1 else 0.0,
            float(d["c_infer"]),
            float(d["h"]),
            float(d["d"]),
            float(d["s"]),
            float(env.get_T()),
        ],
        dtype=float,
    )


def one_step_utility(env: StepCountJITAI, action: int, assumed_c: int | None = None) -> float:
    """
    Counterfactual one-step reward under an assumed context.
    If assumed_c is None, use the env's current true context (oracle).
    Restores env state after the probe.
    """
    snap = snapshot(env)
    try:
        if assumed_c is not None:
            set_true_context(env, int(assumed_c))
        _, reward, _, _, _ = env.step(int(action))
        return float(reward)
    finally:
        restore(env, snap)


def utilities_all_actions(env: StepCountJITAI, assumed_c: int | None = None) -> np.ndarray:
    n = int(env.action_space.n)
    return np.array([one_step_utility(env, a, assumed_c=assumed_c) for a in range(n)], dtype=float)
