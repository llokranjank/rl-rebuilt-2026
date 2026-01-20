import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim.gym_env import LogisticsEnv
from stable_baselines3 import PPO
import os
import shutil

def train():
    # optimized hyperparameters
    env = LogisticsEnv()
    
    # Paths
    current_model_path = "ppo_logistics_v21_3v3"
    opponent_path = "ppo_logistics_opponent.zip"
    
    
    # Self-Play Settings (Overnight Run Configuration)
    # Estimated FPS: ~1800
    # Steps Per Gen: 1,000,000 (~10 minutes)
    # Generations: 50 (~8.5 hours total)

    # Short run for testing
    # GENERATIONS = 5
    # STEPS_PER_GEN = 100000
    
    GENERATIONS = 25
    STEPS_PER_GEN = 5000000
    
    # Initialize Model
    if os.path.exists(current_model_path + ".zip"):
         print("Loading existing model...")
         model = PPO.load(current_model_path, env=env, tensorboard_log="./ppo_logistics_tensorboard/")
    else:
         print("Creating new model...")
         model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logistics_tensorboard/")
    
    print(f"Starting Self-Play Training ({GENERATIONS} Generations)...")
    
    for gen in range(1, GENERATIONS + 1):
        print(f"\n--- Generation {gen}/{GENERATIONS} ---")
        
        # 1. Ensure Opponent exists (Copy current best to opponent slot)
        # For the very first run, if no model exists, Blue is random (handled by Env)
        # After Gen 1, we save Gen 1 as opponent for Gen 2.
        if os.path.exists(current_model_path + ".zip"):
            shutil.copy(current_model_path + ".zip", opponent_path)
            print("Updated Opponent to latest model.")
            # Reload env to pick up new opponent
            env.reset(options={"reload_opponent": True})
            
        # 2. Train
        model.learn(total_timesteps=STEPS_PER_GEN, reset_num_timesteps=False)
        
        # 3. Save
        model.save(current_model_path)
        print(f"Generation {gen} Complete. Model Saved.")
        
    print("Training Cycle Complete.")

if __name__ == "__main__":
    train()
