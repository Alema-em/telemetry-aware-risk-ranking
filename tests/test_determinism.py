import numpy as np

from telemetry_ranking.journal.cvar_rankers import rank_map_sim
from telemetry_ranking.rank.env_utils import make_env
from telemetry_ranking.rank.rankers import policy_regret


def episode(seed: int) -> tuple[float, float, int]:
    env = make_env(sigma=1.0, seed=seed, n_version=0)
    env.reset()
    env.rng0 = np.random.default_rng(seed + 1)
    env.rng1 = np.random.default_rng(seed + 2)
    done = False
    regret = 0.0
    reward = 0.0
    steps = 0
    while not done:
        action = rank_map_sim(env)
        regret += policy_regret(env, action)
        _, value, terminated, truncated, _ = env.step(action)
        reward += float(value)
        done = bool(terminated or truncated)
        steps += 1
    return regret, reward, steps


def test_fixed_seed_episode_is_deterministic() -> None:
    assert episode(5) == episode(5)
