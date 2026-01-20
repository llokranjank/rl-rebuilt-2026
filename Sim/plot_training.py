import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import numpy as np

def plot_training_curve(log_dir, save_path):
    # Find all runs
    # Runs are folders PPO_1, PPO_2, ...
    folders = sorted(glob.glob(os.path.join(log_dir, "PPO_*")), key=os.path.getmtime)
    if not folders:
        print("No training logs found.")
        return
        
    print(f"Found {len(folders)} training sessions. Aggregating...")
    
    plt.figure(figsize=(12, 6))
    
    total_steps = 0
    all_steps = []
    all_rewards = []
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(folders)))
    
    for i, folder in enumerate(folders):
        try:
            event_acc = EventAccumulator(folder)
            event_acc.Reload()
            events = event_acc.Scalars('rollout/ep_rew_mean')
            
            steps = [e.step + total_steps for e in events]
            vals = [e.value for e in events]
            
            if not steps: continue
            
            label = f"Gen {i+1}"
            plt.plot(steps, vals, label=label, color=colors[i], linewidth=2)
            
            # Track continuous line
            all_steps.extend(steps)
            all_rewards.extend([v for v in vals]) # Flatten? No, vals is list
            
            max_step = events[-1].step
            total_steps += max_step
            
        except Exception as e:
            print(f"Skipping {folder}: {e}")

    # Plot trend line
    if all_steps:
        # smooth
        from scipy.signal import savgol_filter
        if len(all_rewards) > 11:
            try:
                yhat = savgol_filter(all_rewards, 11, 3)
                plt.plot(all_steps, yhat, 'r--', alpha=0.5, label='Trend')
            except:
                pass

    plt.xlabel("Total Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("RL Training Progress (Self-Play Generations)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path)
    plt.close() # Close to free memory
    print(f"Saved training curve to {save_path}")

if __name__ == "__main__":
    plot_training_curve("./ppo_logistics_tensorboard/", "training_curve.png")
