import numpy as np

from telemetry_ranking.journal.telemetry_loss import TelemetryChannel


def channel_trace(kind: str, seed: int, steps: int = 200) -> list[bool]:
    channel = TelemetryChannel(kind, 0.2, np.random.default_rng(seed))
    channel.reset()
    return [channel.delivered() for _ in range(steps)]


def test_independent_loss_is_repeatable() -> None:
    assert channel_trace("independent", 17) == channel_trace("independent", 17)


def test_bursty_loss_is_repeatable_and_clustered() -> None:
    trace = channel_trace("bursty", 23)
    drops = [not delivered for delivered in trace]
    adjacent_drops = sum(a and b for a, b in zip(drops, drops[1:]))

    assert trace == channel_trace("bursty", 23)
    assert adjacent_drops > 0


def test_long_run_loss_rate_matches_target() -> None:
    channel = TelemetryChannel(
        "bursty",
        0.2,
        np.random.default_rng(31),
        mean_burst_len=3.0,
    )
    channel.reset()
    for _ in range(20_000):
        channel.delivered()

    assert abs(channel.empirical_loss - 0.2) < 0.03
