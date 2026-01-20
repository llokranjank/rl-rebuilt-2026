
# Simulation Constants
TOTAL_BALLS = 504
MAX_INTAKE_RATE = 6.0  # balls per second

# Timing
MATCH_DURATION = 160 # 20s Auto + 140s Teleop
AUTO_DURATION = 20
# Teleop Shifts
SHIFT_START_TIME = 30
SHIFT_DURATION = 25

# Zone Keys
ZONE_NEUTRAL = "Neutral"
ZONE_RED = "Red"
ZONE_BLUE = "Blue"

# Hub Status
HUB_ACTIVE = "ACTIVE"
HUB_INACTIVE = "INACTIVE"

# --- Configuration Management ---
import os
import yaml

def load_config(yaml_path):
    """Loads configuration from a YAML file and updates global variables."""
    global ROBOT_CONFIGS, TRAINING_CONFIG
    
    if os.path.exists(yaml_path):
        print(f"Loading config from: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Parse Robots
        if "robots" in data:
            ROBOT_CONFIGS = data["robots"]
        
        # Parse Experiment
        if "experiment" in data:
            TRAINING_CONFIG = {
                "EXPERIMENT_NAME": data["experiment"].get("name", "Default"),
                "GENERATIONS": data["experiment"].get("generations", 5),
                "STEPS_PER_GEN": data["experiment"].get("steps_per_gen", 10000),
                "OUTPUT_DIR": data["experiment"].get("output_dir", "./experiments"),
                "SAVE_INTERVAL_GEN": data["experiment"].get("save_interval_gen", 1),
                "ANALYSIS_INTERVAL_GEN": data["experiment"].get("analysis_interval_gen", 1),
            }
    else:
        print(f"WARNING: Config file {yaml_path} not found. Using defaults.")

# Defaults (Overwritten if load_config is called)
ROBOT_CONFIGS = [
    {'capacity': 30, 'transit_time': 2.0, 'balls_shot_per_sec': 8.0, 'intake_efficiency': 1.0},
    {'capacity': 40, 'transit_time': 2.0, 'balls_shot_per_sec': 6.0, 'intake_efficiency': 1.0},
    {'capacity': 60, 'transit_time': 3.0, 'balls_shot_per_sec': 5.0, 'intake_efficiency': 1.0}
]

TRAINING_CONFIG = {
    "EXPERIMENT_NAME": "Default_Exp",
    "GENERATIONS": 5,
    "STEPS_PER_GEN": 10000,
    "OUTPUT_DIR": "./experiments",
    "SAVE_INTERVAL_GEN": 1,
    "ANALYSIS_INTERVAL_GEN": 1
}

# Auto-load default if present AND main script hasn't overridden it yet?
# For backward compatibility, check for a default yaml if no explicit load happens?
# Better: engine.py/gym_env.py just use what's in ROBOT_CONFIGS.
# run_experiment.py will call load_config() explicitly.
# If running gym_env.py directly (rare), it uses defaults.
