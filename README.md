# 2026 Robotics Simulation & RL Environment

This project is a Reinforcement Learning (RL) simulation environment designed to optimize logistics and strategy for a robotics game (likely FRC-themed). It simulates a 3v3 match where robots move between zones (Red, Neutral, Blue), intake balls, pass them, and shoot into hubs.

The agent uses Proximal Policy Optimization (PPO) from `stable-baselines3` and trains via self-play.

## Installation

Ensure you have Python 3.8+ installed.

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:llokranjank/rl-rebuilt-2026.git
    cd rl-rebuilt-2026/Sim
    ```

2.  **Install dependencies:**
    ```bash
    pip install stable-baselines3 shimmy gymnasium matplotlib numpy pyyaml
    ```
    *(Note: You may need `gym` or `gymnasium` depending on the exact versions in `gym_env.py`. If `gym_env.py` uses `gym`, install `gym`.)*

## Usage

### 1. Training (Basic)
To run a simple training loop with hardcoded parameters:
```bash
python train_rl.py
```
This script runs a self-play loop where the agent trains against previous versions of itself.

### 2. Running Experiments (Advanced)
To run a full experiment with configurable parameters, logging, and automatic analysis:
```bash
python run_experiment.py --config experiment_configs/defaultExperiment.yaml
```
- This creates a new directory in `experiments/` with the current timestamp.
- It saves model checkpoints, training curves, and match visualizations.

### 3. Visualizing a Match
To generate a GIF animation of a match using the latest trained model:
```bash
python visualize_match.py
```
This will output `match_simulation.gif`.

## Configuration

Experiments are configured via YAML files in `experiment_configs/`.
Example `defaultExperiment.yaml`:

```yaml
experiment:
  name: "My_Experiment"
  generations: 30
  steps_per_gen: 100000 

robots:
  - id: 1
    type: "Speedy"
    capacity: 20
    transit_time: 1.5
    ...
```

## Structure

- **`gym_env.py`**: Custom Gymnasium environment defining the game logic, observation space, and action space.
- **`engine.py`**: The core simulation engine handling robot physics, scoring, and state updates.
- **`train_rl.py`**: Basic training script.
- **`run_experiment.py`**: Advanced experiment runner.
- **`visualize_match.py`**: Visualization tool using Matplotlib.
- **`run_analysis.py`**: Script to generate plots and statistics from match data.
