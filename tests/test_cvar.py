import numpy as np

from telemetry_ranking.journal.cvar_rankers import rank_from_scores, score_cvar


def test_cvar_prefers_robust_action() -> None:
    probs = np.array([0.8, 0.2])
    utilities = np.array([[10.0, 5.0], [0.0, 5.0]])
    scores = score_cvar(probs, utilities, alpha=0.3)

    assert np.isclose(scores[0], 10.0 / 3.0)
    assert np.isclose(scores[1], 5.0)
    assert rank_from_scores(scores) == 1


def test_cvar_at_alpha_one_matches_mean() -> None:
    probs = np.array([0.6, 0.4])
    utilities = np.array([[8.0, 1.0], [2.0, 7.0]])

    assert np.allclose(score_cvar(probs, utilities, alpha=1.0), probs @ utilities)
