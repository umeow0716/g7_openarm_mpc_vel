# g7_openarm_mpc

A simple velocity-based MPC project for the G7 OpenArm mobile manipulator.
Task for dual arm end effector target tracking

> Note: A low-level controller has not been added yet.
> Just control mujoco qvel directly for PoC controller effect.

## Project Structure

- `src/` - core MPC and robot-related logic
  - `mpc_solver.py` - main MPC solver
  - `mpc_types.py` - shared data types
  - `openarm_dynamic.py` - robot dynamics
  - `openarm_idx.py` - state and index definitions
  - `openarm_limit.py` - joint and motion limits
  - `pinnzoo.py` / `pinnzoo_binding.py` - robot model bindings
  - `utils.py` - utility functions
- `sim_viewer.py` - main simulation entry point with MuJoCo
- `plotter.py` - realtime plotting and target/state visualization (In sim_viewer.py import it)

## Installation

### conda or venv

```bash
# create and source virtual environment first
pip install -r requirements.txt
python sim_viewer.py
```

### uv

```bash
uv sync
uv run sim_viewer.py
```