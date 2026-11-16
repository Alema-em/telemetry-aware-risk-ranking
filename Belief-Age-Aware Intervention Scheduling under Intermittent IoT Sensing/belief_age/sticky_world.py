from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from belief_age.age_gate import sticky_transition

def _score_map(probs: np.ndarray, U: np.ndarray) -> np.ndarray:
    c_hat = int(np.argmax(probs))
    return U[c_hat]

def _score_cvar(probs: np.ndarray, U: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Discrete CVaR over contexts."""
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    p = probs / max(float(probs.sum()), 1e-12)
    _C, A = U.shape
    scores = np.zeros(A, dtype=float)
    for a in range(A):
        utils = U[:, a]
        order = np.argsort(utils)
        u_sorted = utils[order]
        p_sorted = p[order]
        remaining = alpha
        cvar_mass = 0.0
        cvar_sum = 0.0
        for u_i, p_i in zip(u_sorted, p_sorted):
            if remaining <= 0:
                break
            take = min(float(p_i), remaining)
            cvar_sum += take * float(u_i)
            cvar_mass += take
            remaining -= take
        scores[a] = cvar_sum / max(cvar_mass, 1e-12)
    return scores

@dataclass
class StickyConfig:
    stay: float = 0.9
    H: float = 50.0
    M: float = 12.5
    L: float = 0.1
    noise_std: float = 0.25
    # observation flip base; effective flip ≈ clip(σ / (σ + σ0), …)
    sigma0: float = 1.0

def build_U(cfg: StickyConfig, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(0.0, cfg.noise_std, size=(2, 4))
    base = np.array(
        [
            [cfg.L, cfg.M, cfg.H, cfg.L],
            [cfg.L, cfg.M, cfg.L, cfg.H],
        ],
        dtype=float,
    )
    return base + n

def observation_flip_prob(sigma: float, cfg: StickyConfig) -> float:
    """Map noise knob σ → Bernoulli flip probability for soft context observation."""
    s = max(float(sigma), 0.0)
    return float(np.clip(s / (s + cfg.sigma0), 0.02, 0.49))

@dataclass
class StickyWorld:
    """Minimal edge-service twin: context, belief, utilities — no Gym API."""

    sigma: float
    seed: int = 0
    cfg: StickyConfig | None = None

    def __post_init__(self) -> None:
        self.cfg = self.cfg or StickyConfig()
        self.rng = np.random.default_rng(self.seed)
        self.M = sticky_transition(self.cfg.stay)
        self.U = build_U(self.cfg, self.rng)
        self.t = 0
        self.c = int(self.rng.integers(0, 2))
        self.belief = np.array([0.5, 0.5], dtype=float)
        self.flip = observation_flip_prob(self.sigma, self.cfg)
        self._refresh_belief_from_obs()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self.U = build_U(self.cfg, self.rng)  # type: ignore[arg-type]
        self.t = 0
        self.c = int(self.rng.integers(0, 2))
        self.belief = np.array([0.5, 0.5], dtype=float)
        self.flip = observation_flip_prob(self.sigma, self.cfg)  # type: ignore[arg-type]
        self._refresh_belief_from_obs()

    def predict_belief(self, steps: int = 1) -> None:
        """Diffuse belief through sticky transitions (open-loop age prediction)."""
        for _ in range(max(int(steps), 0)):
            self.belief = self.belief @ self.M
            self.belief = self.belief / max(float(self.belief.sum()), 1e-12)

    def _refresh_belief_from_obs(self) -> None:
        """Bayes update: sticky predictive prior × soft observation likelihood."""
        obs = self.c if self.rng.random() >= self.flip else 1 - self.c
        like = np.array(
            [1.0 - self.flip if obs == 0 else self.flip, 1.0 - self.flip if obs == 1 else self.flip],
            dtype=float,
        )
        prior = self.belief @ self.M
        prior = prior / max(float(prior.sum()), 1e-12)
        post = prior * like
        self.belief = post / max(float(post.sum()), 1e-12)

    def belief_probs(self) -> np.ndarray:
        return self.belief.copy()

    def utilities(self) -> np.ndarray:
        return self.U.copy()

    def oracle_action(self) -> int:
        return int(np.argmax(self.U[self.c]))

    def policy_regret(self, action: int) -> float:
        best = float(np.max(self.U[self.c]))
        got = float(self.U[self.c, int(action)])
        return best - got

    def choose_map(self, probs: np.ndarray | None = None) -> int:
        p = self.belief if probs is None else np.asarray(probs, dtype=float)
        return int(np.argmax(_score_map(p, self.U)))

    def choose_cvar(self, probs: np.ndarray | None = None, alpha: float = 0.3) -> int:
        p = self.belief if probs is None else np.asarray(probs, dtype=float)
        return int(np.argmax(_score_cvar(p, self.U, alpha=alpha)))

    def step(self, action: int, monitoring_delivered: bool, predict_on_miss: bool = True) -> dict:
        """Apply intervention, evolve context, optionally refresh belief."""
        reward = float(self.U[self.c, int(action)])
        regret = self.policy_regret(action)
        # sticky transition of latent context
        self.c = int(self.rng.choice(2, p=self.M[self.c]))
        self.t += 1
        if monitoring_delivered:
            self._refresh_belief_from_obs()
        elif predict_on_miss:
            # Open-loop predict: gateway belief ages with the sticky chain
            self.predict_belief(1)
        # else: frozen belief (Always-Act-on-frozen-belief baseline)
        return {
            "reward": reward,
            "regret": regret,
            "c": self.c,
            "belief": self.belief.copy(),
            "t": self.t,
        }

def _finite_or_nan(v) -> float:
    if v is None:
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")

def ingest_telemetry_proxy(world: StickyWorld, telemetry: dict) -> dict:
    """Map live MQTT fields into a soft belief refresh (hardware path).

    Missing / null sensor fields stay NaN — never invent HR or other readings.
    """
    hr = _finite_or_nan(telemetry.get("hr"))
    steps = _finite_or_nan(telemetry.get("steps"))
    activity = _finite_or_nan(telemetry.get("activity"))
    temp_c = telemetry.get("temp_c")
    sensors = telemetry.get("sensors") or {}
    source = sensors.get("source", "hardware")

    # Prefer real IMU activity; use HR only when a valid reading exists
    active = False
    if np.isfinite(activity):
        active = activity > 1.5
    elif np.isfinite(hr):
        active = hr > 85.0
    elif np.isfinite(steps):
        active = (steps % 50) > 25
    if temp_c is not None:
        try:
            if float(temp_c) > 28.0:
                active = True
        except Exception:
            pass

    obs = 1 if active else 0
    flip = world.flip
    like = np.array(
        [1.0 - flip if obs == 0 else flip, 1.0 - flip if obs == 1 else flip],
        dtype=float,
    )
    prior = world.belief.copy()
    world.c = int(world.rng.choice(2, p=world.M[world.c]))
    post = prior * like
    world.belief = post / max(float(post.sum()), 1e-12)
    world.t += 1
    return {
        "belief": world.belief.tolist(),
        "features": [
            float(world.belief[0]),
            float(world.belief[1]),
            hr,
            steps if np.isfinite(steps) else 0.0,
            activity if np.isfinite(activity) else 0.0,
            float(temp_c) if temp_c is not None else float("nan"),
        ],
        "twin_steps_ingested": world.t,
        "c_hint": obs,
        "sensor_source": source,
        "sensors": sensors,
    }
