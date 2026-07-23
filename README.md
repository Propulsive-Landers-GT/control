# control

Vehicle-agnostic **control algorithms** for GT Propulsive Landers — the controller
designs and studies that can be tuned and dropped into any GTPL vehicle.

## Contents

| Path | What it is |
|------|------------|
| `LQR/` | Linear-quadratic regulator gain design (MATLAB) |
| `Controllers/` | `LQR.py` — Python LQR controller |
| `MPC/`, `MPC2/` | Model-predictive control (Python + Rust) |
| `TVC/` | Thrust-vector control (MATLAB, C++, Python) |
| `RCS/` | Reaction-control-system control (MATLAB + Rust) |
| `ReactionControlSystem/` | RCS stabilization studies (MATLAB) |
| `MTV Sim/` | PID tuning simulation |

> **Note:** The `LQR/` (design) here is distinct from the integrated C++ runtime
> LQR that lives in [`monoprop-flight-software`](../monoprop-flight-software)'s
> `ControlLoop/`. This repo is the algorithm library; that one is the vehicle build.

## Sibling libraries

[`navigation`](../navigation) · [`guidance`](../guidance) · [`simulations`](../simulations)
