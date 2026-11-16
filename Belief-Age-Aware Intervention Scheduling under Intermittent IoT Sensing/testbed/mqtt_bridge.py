from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from belief_age.age_gate import BeliefAgeTracker, GateConfig, belief_peak, decide_gate
from belief_age.sticky_world import StickyWorld, ingest_telemetry_proxy

ACTION_NAMES = {
    0: "no_intervention",
    1: "low_intensity_nudge",
    2: "context0_specialized",
    3: "context1_specialized",
}


class Twin:
    def __init__(self, sigma: float = 1.0, seed: int = 0):
        self.world = StickyWorld(sigma=sigma, seed=seed)
        self.world.reset(seed=seed)
        self.tracker = BeliefAgeTracker()
        self.cfg = GateConfig(policy="peak_and_age")
        self.msgs = 0

    def on_packet(self, telemetry: dict[str, Any], delivered: bool = True) -> dict[str, Any]:
        if delivered:
            ingest_telemetry_proxy(self.world, telemetry)
            self.tracker.on_update(self.world.belief_probs())
        else:
            self.tracker.on_stale()
            self.world.predict_belief(1)
        self.msgs += 1
        p = self.world.belief_probs()
        gate = decide_gate(self.tracker.delta, belief_peak(p), self.cfg)
        if gate.value == "act":
            action = self.world.choose_map(p)
        elif gate.value == "defer":
            action = 1
        else:
            action = 0
        return {
            "seq": telemetry.get("seq"),
            "delta": self.tracker.delta,
            "gate": gate.value,
            "action": int(action),
            "action_name": ACTION_NAMES[int(action)],
            "belief": p.tolist(),
            "msgs": self.msgs,
        }


def run_sim(n: int = 40, duty: int = 3) -> None:
    twin = Twin()
    for i in range(n):
        delivered = (i % duty) == 0
        tel = {
            "device_id": "sim",
            "seq": i,
            "t_ms": i * 1000,
            "steps": float(i),
            "hr": 72.0,
            "activity": 1.0 + (i % 5),
        }
        print(json.dumps(twin.on_packet(tel, delivered=delivered)))


def run_live(host: str, topic: str) -> None:
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise SystemExit("paho-mqtt is required for --live") from exc

    twin = Twin()

    def on_message(_client, _userdata, msg):
        try:
            tel = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        print(json.dumps(twin.on_packet(tel, delivered=True)), flush=True)

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(host, 1883, 60)
    client.subscribe(topic)
    client.loop_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--topic", default="jitai/telemetry/#")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    if args.live:
        run_live(args.host, args.topic)
    else:
        run_sim(n=args.n)


if __name__ == "__main__":
    main()
