# control

Vehicle-agnostic **control algorithms** for GT Propulsive Landers — the controller
designs and studies that can be tuned and dropped into any GTPL vehicle.

> **Provenance:** Reorganized **by function** in July 2026 from the monolithic
> [`Propulsive-Landers-GT/MonopropUAV`](https://github.com/Propulsive-Landers-GT/MonopropUAV)
> repo. These were previously scattered across `Algorithms/` and
> `Sensing&Controls/Controllers/`. Full commit history is preserved. See the
> [GTPL-test root README](../README.md) for the full mapping.

## Contents

| Path | What it is | Was |
|------|------------|-----|
| `LQR/` | Linear-quadratic regulator gain design (MATLAB) | `Algorithms/LQR` |
| `Controllers/` | `LQR.py` — Python LQR controller | `Sensing&Controls/Controllers` |
| `MPC/`, `MPC2/` | Model-predictive control (Python + Rust) | `Algorithms/MPC`, `MPC2` |
| `TVC/` | Thrust-vector control (MATLAB, C++, Python) | `Algorithms/TVC` |
| `RCS/` | Reaction-control-system control (MATLAB + Rust) | `Algorithms/RCS` |
| `ReactionControlSystem/` | RCS stabilization studies (MATLAB) | `Algorithms/ReactionControlSystem` |
| `MTV Sim/` | PID tuning simulation | `Algorithms/MTV Sim` |

> **Note:** The `LQR/` (design) here is distinct from the integrated C++ runtime
> LQR that lives in [`monoprop-flight-software`](../monoprop-flight-software)'s
> `ControlLoop/`. This repo is the algorithm library; that one is the vehicle build.

## Sibling libraries

[`navigation`](../navigation) · [`guidance`](../guidance) · [`dynamics`](../dynamics)
