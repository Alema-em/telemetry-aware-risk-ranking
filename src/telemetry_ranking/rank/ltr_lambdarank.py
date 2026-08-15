"""LightGBM LambdaRank for belief-aware / MAP-ablation intervention ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np

from telemetry_ranking.rank.env_utils import belief_probs, context_features, make_env, utilities_all_actions

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

FeatureMode = Literal["belief", "map", "residual"]
N_ACTIONS = 4


def belief_feature_vector(feats: np.ndarray) -> np.ndarray:
    """Full belief/state features: [p0, p1, L, H, D, S, T]."""
    return np.asarray(feats, dtype=float).reshape(-1)


def map_feature_vector(feats: np.ndarray) -> np.ndarray:
    """
    Ablation: drop soft belief; keep hard MAP context as one-hot-ish signal.
    Replace p0,p1 with (1,0) or (0,1) according to argmax of original probs,
    keep L,H,D,S,T.
    """
    f = belief_feature_vector(feats)
    c_hat = int(np.argmax(f[:2]))
    out = f.copy()
    out[0] = 1.0 if c_hat == 0 else 0.0
    out[1] = 1.0 if c_hat == 1 else 0.0
    return out


def action_one_hot(action: int, n: int = N_ACTIONS) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    v[int(action)] = 1.0
    return v


def map_sim_utilities(env) -> np.ndarray:
    """One-step utilities under MAP-collapsed context (same as B1_MAP)."""
    probs = belief_probs(env)
    c_hat = int(np.argmax(probs))
    return utilities_all_actions(env, assumed_c=c_hat)


def residual_row_features(
    feats: np.ndarray,
    action: int,
    map_utils: np.ndarray,
) -> np.ndarray:
    """Belief/state + action id + full MAP-sim utility vector + this action's MAP util."""
    base = belief_feature_vector(feats)
    mu = np.asarray(map_utils, dtype=float).reshape(-1)
    return np.concatenate([base, action_one_hot(action), mu, [float(mu[int(action)])]])


def row_features(
    feats: np.ndarray,
    action: int,
    mode: FeatureMode,
    map_utils: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "residual":
        if map_utils is None:
            raise ValueError("residual mode requires map_utils")
        return residual_row_features(feats, action, map_utils)
    base = belief_feature_vector(feats) if mode == "belief" else map_feature_vector(feats)
    return np.concatenate([base, action_one_hot(action)])


def relevance_from_utils(utils: np.ndarray) -> np.ndarray:
    """Integer relevance labels for LambdaRank (higher = better)."""
    order = np.argsort(np.argsort(utils))  # ranks ascending
    # map to 0..n-1 with best getting highest
    return order.astype(int)


def collect_listwise(
    n_queries: int = 4000,
    sigmas: list[float] | None = None,
    seed: int = 0,
) -> list[dict]:
    if sigmas is None:
        sigmas = [0.0, 0.4, 1.0, 2.0]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    attempts = 0
    max_attempts = n_queries * 30

    while len(rows) < n_queries and attempts < max_attempts:
        attempts += 1
        sigma = float(rng.choice(sigmas))
        ep_seed = int(rng.integers(0, 1_000_000))
        env = make_env(sigma=sigma, seed=ep_seed, n_version=0)
        env.reset()
        for _ in range(int(rng.integers(0, 8))):
            a = int(rng.integers(0, env.action_space.n))
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset()

        feats = context_features(env)
        utils = utilities_all_actions(env, assumed_c=None)
        if float(np.ptp(utils)) < 1e-6:
            continue
        map_utils = map_sim_utilities(env)
        rows.append(
            {
                "query_id": len(rows),
                "sigma": sigma,
                "features": feats.tolist(),
                "utils": [float(u) for u in utils],
                "map_utils": [float(u) for u in map_utils],
                "relevance": relevance_from_utils(utils).tolist(),
                "best_action": int(np.argmax(utils)),
            }
        )
    return rows


def build_lgb_matrix(rows: list[dict], mode: FeatureMode) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[int] = []
    for r in rows:
        feats = np.asarray(r["features"], dtype=float)
        rel = r["relevance"]
        map_utils = np.asarray(r["map_utils"], dtype=float) if mode == "residual" else None
        for a in range(N_ACTIONS):
            xs.append(row_features(feats, a, mode, map_utils=map_utils))
            ys.append(int(rel[a]))
        groups.append(N_ACTIONS)
    return np.vstack(xs), np.asarray(ys, dtype=int), groups


def build_residual_regression_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Targets = oracle utility minus MAP-sim utility (residual to learn)."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for r in rows:
        feats = np.asarray(r["features"], dtype=float)
        oracle_u = np.asarray(r["utils"], dtype=float)
        map_u = np.asarray(r["map_utils"], dtype=float)
        for a in range(N_ACTIONS):
            xs.append(residual_row_features(feats, a, map_u))
            ys.append(float(oracle_u[a] - map_u[a]))
    return np.vstack(xs), np.asarray(ys, dtype=float)


def _split_query_indices(n_q: int, seed: int = 0) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_q)
    n_val = max(1, n_q // 10)
    val_q = set(int(i) for i in perm[:n_val])
    train_q = set(range(n_q)) - val_q
    return train_q, val_q


def train_lambdarank(
    rows: list[dict],
    mode: FeatureMode,
    out_path: Path,
    seed: int = 0,
) -> lgb.Booster:
    x, y, groups = build_lgb_matrix(rows, mode)
    n_q = len(rows)
    train_q, val_q = _split_query_indices(n_q, seed=seed)

    train_idx: list[int] = []
    val_idx: list[int] = []
    train_groups: list[int] = []
    val_groups: list[int] = []
    for qi in range(n_q):
        sl = slice(qi * N_ACTIONS, (qi + 1) * N_ACTIONS)
        if qi in val_q:
            val_idx.extend(range(sl.start, sl.stop))
            val_groups.append(N_ACTIONS)
        else:
            train_idx.extend(range(sl.start, sl.stop))
            train_groups.append(N_ACTIONS)

    dtrain = lgb.Dataset(x[train_idx], label=y[train_idx], group=train_groups)
    dval = lgb.Dataset(x[val_idx], label=y[val_idx], group=val_groups, reference=dtrain)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [1, 4],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "seed": seed,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path))
    return booster


def train_residual_ltr(
    rows: list[dict],
    out_path: Path,
    seed: int = 0,
) -> lgb.Booster:
    """
    Learn delta so final score = MAP-sim utility + delta(belief, action, map utilities).
    Smallest learning-based tweak: residual correction of MAP under belief.
    """
    if any("map_utils" not in r for r in rows):
        raise ValueError("residual training requires map_utils in each listwise row")

    x, y = build_residual_regression_matrix(rows)
    n_q = len(rows)
    train_q, val_q = _split_query_indices(n_q, seed=seed)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for qi in range(n_q):
        sl = slice(qi * N_ACTIONS, (qi + 1) * N_ACTIONS)
        if qi in val_q:
            val_idx.extend(range(sl.start, sl.stop))
        else:
            train_idx.extend(range(sl.start, sl.stop))

    dtrain = lgb.Dataset(x[train_idx], label=y[train_idx])
    dval = lgb.Dataset(x[val_idx], label=y[val_idx], reference=dtrain)
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "seed": seed,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path))
    return booster


def load_booster(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def score_actions(env, booster: lgb.Booster, mode: FeatureMode) -> np.ndarray:
    feats = context_features(env)
    map_utils = map_sim_utilities(env) if mode == "residual" else None
    x = np.vstack([row_features(feats, a, mode, map_utils=map_utils) for a in range(N_ACTIONS)])
    pred = np.asarray(booster.predict(x), dtype=float)
    if mode == "residual":
        # final score = MAP utility + learned residual
        return np.asarray(map_utils, dtype=float) + pred
    return pred


def rank_with_booster(env, booster: lgb.Booster, mode: FeatureMode) -> int:
    scores = score_actions(env, booster, mode)
    return int(np.argmax(scores))


def main():
    parser = argparse.ArgumentParser(description="Train Belief-LTR, MAP-LTR and Residual-LTR models")
    parser.add_argument("--queries", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-data", action="store_true", help="Reuse existing listwise jsonl")
    parser.add_argument(
        "--residual-only",
        action="store_true",
        help="Only (re)build residual model; leave Belief/MAP LTR models unchanged",
    )
    args = parser.parse_args()

    listwise_path = DATA / "oracle_listwise_train.jsonl"
    meta_path = DATA / "oracle_listwise_meta.json"

    need_map_utils = True
    if args.skip_data and listwise_path.exists():
        rows = [json.loads(line) for line in listwise_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        need_map_utils = any("map_utils" not in r for r in rows)
        if need_map_utils:
            print("Existing listwise data lacks map_utils; regenerating...")
        else:
            print(f"Loaded {len(rows)} listwise queries from {listwise_path}")

    if not (args.skip_data and listwise_path.exists() and not need_map_utils):
        print(f"Collecting {args.queries} listwise oracle queries (with MAP-sim utilities)...")
        rows = collect_listwise(n_queries=args.queries, seed=args.seed)
        with open(listwise_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        meta = {
            "n_queries": len(rows),
            "n_actions": N_ACTIONS,
            "feature_dim_context": 7,
            "feature_dim_row_belief": 7 + N_ACTIONS,
            "feature_dim_row_residual": 7 + N_ACTIONS + N_ACTIONS + 1,
            "sigmas": sorted({r["sigma"] for r in rows}),
            "path": str(listwise_path),
            "has_map_utils": True,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {len(rows)} queries -> {listwise_path}")

    residual_path = MODELS / "residual_ltr.txt"
    if not args.residual_only:
        belief_path = MODELS / "belief_ltr.txt"
        map_path = MODELS / "map_ltr.txt"
        print("Training Belief-LTR...")
        train_lambdarank(rows, "belief", belief_path, seed=args.seed)
        print(f"Saved {belief_path}")
        print("Training MAP-LTR (ablation)...")
        train_lambdarank(rows, "map", map_path, seed=args.seed)
        print(f"Saved {map_path}")

    print("Training Residual-LTR (MAP-sim + learned belief residual)...")
    train_residual_ltr(rows, residual_path, seed=args.seed)
    print(f"Saved {residual_path}")


if __name__ == "__main__":
    main()
