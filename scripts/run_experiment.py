import os
import shutil
import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim import config
from sim.gym_env import LogisticsEnv
from stable_baselines3 import PPO
from plot_training import plot_training_curve
from visualize_match import run_and_animate
from run_analysis import run_analysis

import argparse

def run_experiment():
    # 0. Parse CLI Arguments
    parser = argparse.ArgumentParser(description="Run Logistics Simulation Experiment")
    parser.add_argument("-c", "--config", type=str, default="configs/defaultExperiment.yaml", help="Path to experiment config YAML")
    args = parser.parse_args()
    
    # Load Config
    # If the default file doesn't exist, config.py will warn and use defaults.
    config.load_config(args.config)

    # 1. Setup Experiment Directory
    cfg = config.TRAINING_CONFIG
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg["EXPERIMENT_NAME"]
    exp_dir = os.path.join(cfg["OUTPUT_DIR"], f"{exp_name}_{timestamp}")
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    print(f"--- Starting Experiment: {exp_name} ---")
    print(f"Config File: {args.config}")
    print(f"Output Directory: {exp_dir}")
    
    # 2. Backup Config
    if os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(exp_dir, "config_snapshot.yaml"))
    
    # 3. Initialize Environment & Model
    env = LogisticsEnv()
    log_dir = os.path.join(exp_dir, "tensorboard_logs")
    
    # Check for existing latest model to resume? 
    # For now, let's assume fresh start or user manually places file.
    # Actually, config says 'EXPERIMENT_NAME', maybe we start fresh.
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # Self-Play Setup
    opponent_path = "ppo_logistics_opponent.zip"
    latest_model_path = os.path.join(exp_dir, "model_latest")
    
    # 4. Training Loop
    generations = cfg["GENERATIONS"]
    steps_per_gen = cfg["STEPS_PER_GEN"]
    
    for gen in range(1, generations + 1):
        print(f"\ncpp--- Generation {gen}/{generations} ---")
        
        # Self-Play Update: Copy latest model to opponent slot
        if os.path.exists(latest_model_path + ".zip"):
            shutil.copy(latest_model_path + ".zip", opponent_path)
            print("Updated Opponent to latest model.")
            env.reset(options={"reload_opponent": True})
        
        # Train
        model.learn(total_timesteps=steps_per_gen, reset_num_timesteps=False)
        
        # Save Latest
        model.save(latest_model_path)
        
        # Save Checkpoint
        if gen % cfg["SAVE_INTERVAL_GEN"] == 0:
            ckpt_path = os.path.join(exp_dir, f"model_gen_{gen}")
            model.save(ckpt_path)
            print(f"Saved Checkpoint: {ckpt_path}")
            
        # Analysis
        if gen % cfg["ANALYSIS_INTERVAL_GEN"] == 0:
            print("Running Analysis...")
            
            # Training Curve
            plot_training_curve(log_dir, os.path.join(exp_dir, f"training_curve_gen_{gen}.png"))
            
            # Match GIF
            run_and_animate(model_path=latest_model_path, 
                            save_path=os.path.join(exp_dir, f"match_simulation_gen_{gen}.gif"),
                            headless=True)
                            
            # Match Analysis PNG
            run_analysis(model_path=latest_model_path,
                         save_path=os.path.join(exp_dir, f"match_analysis_gen_{gen}.png"))
                         
    print("\n--- Experiment Complete ---")
    print(f"All artifacts saved to: {exp_dir}")

if __name__ == "__main__":
    run_experiment()
