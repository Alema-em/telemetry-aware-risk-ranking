# Belief-Age-Aware Intervention Scheduling under Intermittent IoT Sensing

Code for belief-age Act / Wait / Defer scheduling under intermittent IoT sensing.

This folder stands alone. It does not need the ranking package in the repository root.

## Layout

- `belief_age/` gate, sticky-world simulator, loss models, experiment runners
- `testbed/` MQTT bridge and ESP32 sketch

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Experiments

```powershell
python -m belief_age.run_delta_regime
python -m belief_age.run_threshold_sensitivity
python -m belief_age.run_codesign_package
python -m belief_age.plot_delta_regime
python -m belief_age.bench_gateway
```

## Testbed

```powershell
python testbed\mqtt_bridge.py --sim
python testbed\mqtt_bridge.py --live --host 192.168.1.10
```

Flash `testbed/firmware/esp32_mqtt_telemetry.ino`. Copy `secrets.example.h` to `secrets.h` for local Wi-Fi and MQTT. Do not commit `secrets.h`.
