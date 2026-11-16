from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ChannelKind = Literal["independent", "bursty"]


@dataclass
class GilbertElliott:
    p_loss: float
    mean_burst_len: float = 3.0

    def __post_init__(self) -> None:
        p = float(np.clip(self.p_loss, 0.0, 0.999))
        r = 1.0 / max(float(self.mean_burst_len), 1.0)
        h = 0.0 if p <= 0 else (p * r) / max(1.0 - p, 1e-12)
        self.p = p
        self.r = float(np.clip(r, 0.0, 1.0))
        self.h = float(np.clip(h, 0.0, 1.0))


class TelemetryChannel:
    def __init__(
        self,
        kind: ChannelKind,
        p_loss: float,
        rng: np.random.Generator,
        mean_burst_len: float = 3.0,
    ) -> None:
        self.kind = kind
        self.p_loss = float(np.clip(p_loss, 0.0, 1.0))
        self.rng = rng
        self.ge = GilbertElliott(self.p_loss, mean_burst_len=mean_burst_len)
        self.in_bad = False
        self.n_steps = 0
        self.n_drops = 0

    def reset(self) -> None:
        self.in_bad = False
        self.n_steps = 0
        self.n_drops = 0
        if self.kind == "bursty" and self.p_loss > 0:
            self.in_bad = bool(self.rng.random() < self.p_loss)

    def delivered(self) -> bool:
        self.n_steps += 1
        if self.p_loss <= 0:
            return True
        if self.kind == "independent":
            drop = bool(self.rng.random() < self.p_loss)
        else:
            if self.in_bad:
                if self.rng.random() < self.ge.r:
                    self.in_bad = False
            else:
                if self.rng.random() < self.ge.h:
                    self.in_bad = True
            drop = self.in_bad
        if drop:
            self.n_drops += 1
            return False
        return True

    @property
    def empirical_loss(self) -> float:
        return float(self.n_drops / max(self.n_steps, 1))
