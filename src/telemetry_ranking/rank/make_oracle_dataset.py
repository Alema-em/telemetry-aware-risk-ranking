"""Generate pairwise oracle distillation labels from StepCountJITAI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from telemetry_ranking.rank.env_utils import context_features, make_env, utilities_all_actions

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def collect_pairs(
    n_pairs: int = 8000,
    sigmas: list[float] | None = None,
    seed: int = 0,
) -> list[dict]:
    if sigmas is None:
        sigmas = [0.0, 0.4, 1.0, 2.0]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    attempts = 0
    max_attempts = n_pairs * 20

    while len(rows) < n_pairs and attempts < max_attempts:
        attempts += 1
        sigma = float(rng.choice(sigmas))
        ep_seed = int(rng.integers(0, 1_000_000))
        env = make_env(sigma=sigma, seed=ep_seed, n_version=0)
        env.reset()
        # burn-in a few random steps so states are diverse
        for _ in range(int(rng.integers(0, 8))):
            a = int(rng.integers(0, env.action_space.n))
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset()

        feats = context_features(env).tolist()
        utils = utilities_all_actions(env, assumed_c=None)
        # sample a pair of distinct actions
        a_i, a_j = rng.choice(env.action_space.n, size=2, replace=False)
        a_i, a_j = int(a_i), int(a_j)
        if abs(utils[a_i] - utils[a_j]) < 1e-6:
            continue
        label = 1 if utils[a_i] > utils[a_j] else 0
        rows.append(
            {
                "sigma": sigma,
                "features": feats,
                "a_i": a_i,
                "a_j": a_j,
                "u_i": float(utils[a_i]),
                "u_j": float(utils[a_j]),
                "label": label,
                "best_action": int(np.argmax(utils)),
                "utils": [float(u) for u in utils],
            }
        )
    return rows


def pairs_to_listwise(pair_rows: list[dict]) -> list[dict]:
    """Deduplicate pairwise rows into listwise queries using shared features+utils."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in pair_rows:
        key = (tuple(r["features"]), tuple(r["utils"]), float(r["sigma"]))
        if key in seen:
            continue
        seen.add(key)
        utils = np.asarray(r["utils"], dtype=float)
        order = np.argsort(np.argsort(utils)).astype(int)
        out.append(
            {
                "query_id": len(out),
                "sigma": r["sigma"],
                "features": r["features"],
                "utils": r["utils"],
                "relevance": order.tolist(),
                "best_action": int(r["best_action"]),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--also-listwise", action="store_true", help="Also write listwise jsonl from pairs")
    args = parser.parse_args()

    rows = collect_pairs(n_pairs=args.pairs, seed=args.seed)
    out = DATA / "oracle_pairs_train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # also save a compact numpy-friendly summary
    meta = {
        "n_pairs": len(rows),
        "feature_dim": len(rows[0]["features"]) if rows else 0,
        "n_actions": 4,
        "sigmas": sorted({r["sigma"] for r in rows}),
        "path": str(out),
    }
    with open(DATA / "oracle_pairs_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(rows)} pairs -> {out}")
    print(meta)

    if args.also_listwise:
        listwise = pairs_to_listwise(rows)
        lw_out = DATA / "oracle_listwise_from_pairs.jsonl"
        with open(lw_out, "w", encoding="utf-8") as f:
            for r in listwise:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(listwise)} listwise queries -> {lw_out}")


if __name__ == "__main__":
    main()
