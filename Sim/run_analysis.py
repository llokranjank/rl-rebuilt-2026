import numpy as np
import matplotlib.pyplot as plt
import config
from gym_env import LogisticsEnv
from stable_baselines3 import PPO

def run_analysis(model_path="ppo_logistics_v21_3v3", save_path="match_analysis.png"):
    # 1. Setup & Run Simulation
    env = LogisticsEnv()
    try:
        if model_path and os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path)
        else:
            model = None
            print("RL Model not found. Using Random Agent.")
    except:
        print("Error loading model. Using Random Agent.")
        model = None

    obs, _ = env.reset()
    engine = env.engine
    
    print("Running Match Analysis Simulation...")
    done = False
    
    steps = 0
    max_steps = 2000
    
    while not done and steps < max_steps:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        if done or truncated:
            break
            
    print(f"Match Finished. Red: {engine.red_score}, Blue: {engine.blue_score}")

    # 2. Visualization
    # Figure Layout: 8 Rows (Score, ZoneFuel, 6x Robots)
    fig, axes = plt.subplots(8, 1, figsize=(15, 24), sharex=True)
    
    # Shared Helper for Hub Background
    def add_hub_background(ax):
        hub_hist = engine.history_hub_status
        h_times = [h['time'] for h in hub_hist]
        r_active = [1 if h['red'] == config.HUB_ACTIVE else 0 for h in hub_hist]
        b_active = [1 if h['blue'] == config.HUB_ACTIVE else 0 for h in hub_hist]
        
        ax.fill_between(h_times, 0, 1, where=r_active, color='red', alpha=0.1, transform=ax.get_xaxis_transform(), label='Red Hub Active')
        ax.fill_between(h_times, 0, 1, where=b_active, color='blue', alpha=0.1, transform=ax.get_xaxis_transform(), label='Blue Hub Active')

    # --- Row 0: Scores ---
    ax_score = axes[0]
    scores = engine.history_score
    t = [s['time'] for s in scores]
    ax_score.plot(t, [s['red'] for s in scores], 'r-', label='Red Score',  linewidth=2)
    ax_score.plot(t, [s['blue'] for s in scores], 'b-', label='Blue Score', linewidth=2)
    ax_score.set_ylabel("Score")
    ax_score.legend(loc='upper left')
    ax_score.set_title("Match Score (Background = Active Hub)")
    ax_score.grid(True, alpha=0.3)
    add_hub_background(ax_score)

    # --- Row 1: Zone Fuel ---
    ax_zfuel = axes[1]
    ball_hist = engine.history_zone_balls
    t_balls = [b['time'] for b in ball_hist]
    ax_zfuel.plot(t_balls, [b['red'] for b in ball_hist], color='red', linestyle='-', label='Red Zone')
    ax_zfuel.plot(t_balls, [b['neutral'] for b in ball_hist], color='gray', linestyle='-', label='Neutral Zone')
    ax_zfuel.plot(t_balls, [b['blue'] for b in ball_hist], color='blue', linestyle='-', label='Blue Zone')
    ax_zfuel.set_ylabel("Zone Balls")
    ax_zfuel.legend(loc='upper right')
    ax_zfuel.set_title("Zone Fuel Levels")
    ax_zfuel.grid(True, alpha=0.3)
    add_hub_background(ax_zfuel)

    # --- Prepare Robot Data ---
    states = engine.history_robot_states
    robots = {}
    for s in states:
        rid = f"{s['team']} {s['robot_idx']}"
        if rid not in robots: robots[rid] = {'time': [], 'zone': [], 'action': [], 'inventory': [], 'cooldown': []}
        robots[rid]['time'].append(s['time'])
        robots[rid]['zone'].append(s['zone'])
        robots[rid]['action'].append(s['action'])
        robots[rid]['inventory'].append(s['inventory'])
    
    robot_ids = sorted(list(robots.keys())) 
    
    # --- Rows 2-7: Individual Robots (Double Axis) ---
    action_order = ['Idle', 'Moving', 'Intaking', 'Passing', 'Shooting']
    action_map = {a: i for i, a in enumerate(action_order)}
    c_map = {'Idle': 'white', 'Shooting': 'gold', 'Passing': 'orange', 'Moving': 'purple', 'Intaking': 'green'}
    
    hub_hist = engine.history_hub_status
    r_hub_active = np.array([1 if h['red'] == config.HUB_ACTIVE else 0 for h in hub_hist])
    b_hub_active = np.array([1 if h['blue'] == config.HUB_ACTIVE else 0 for h in hub_hist])
    
    for i, rid in enumerate(robot_ids):
        ax = axes[2 + i]
        data = robots[rid]
        times = np.array(data['time'])
        
        # --- Left Axis: Actions ---
        
        # 1. Background Color = Location
        zones = np.array(data['zone'])
        changes = np.where(zones[:-1] != zones[1:])[0]
        change_indices = np.concatenate(([0], changes + 1, [len(zones)]))
        
        for j in range(len(change_indices) - 1):
            start_idx = change_indices[j]
            end_idx = change_indices[j+1]
            z = zones[start_idx]
            t_start = times[start_idx]
            t_end = times[min(end_idx, len(times)-1)]
            
            color = 'white'
            if z == config.ZONE_RED: color = '#ffcccc' 
            elif z == config.ZONE_BLUE: color = '#ccccff' 
            elif z == config.ZONE_NEUTRAL: color = '#eeeeee' 
            
            ax.axvspan(t_start, t_end, facecolor=color, alpha=0.5)

        # 2. Action Scatter
        act_y = []
        act_c = []
        for a in data['action']:
             matched_k = 'Idle'
             for k in action_order:
                 if k in a: matched_k = k; break
             act_y.append(action_map[matched_k])
             act_c.append(c_map[matched_k])
             
        ax.scatter(times, act_y, c=act_c, s=10, marker='|') # Action ticks
        
        # 3. Hub Active Indicator
        is_red_bot = 'Red' in rid
        my_hub_active = r_hub_active if is_red_bot else b_hub_active
        active_indices = np.where(my_hub_active == 1)[0]
        if len(active_indices) > 0:
            active_times = times[active_indices]
            ax.scatter(active_times, [-1] * len(active_times), color='gold', marker='s', s=5)

        ax.set_yticks(list(range(len(action_order))) + [-1])
        ax.set_yticklabels(action_order + ['HubActive'], fontsize=8)
        ax.set_ylim(-1.5, 4.5)
        ax.set_ylabel("Action")
        ax.set_title(f"Robot: {rid}")
        
        # --- Right Axis: Fuel Inventory ---
        ax_inv = ax.twinx()
        fuel_color = 'darkred' if is_red_bot else 'darkblue'
        
        # Determine Capacity for this Robot ID
        # rid format: "Team ID" e.g. "Red 1"
        try:
             bot_id_num = int(rid.split(" ")[1]) # 1, 2, 3
             capacity = config.ROBOT_CONFIGS[bot_id_num-1]['capacity']
        except:
             capacity = 30 # Fallback
             
        ax_inv.plot(times, data['inventory'], color=fuel_color, linewidth=1.5, alpha=0.7, label='Inventory')
        ax_inv.set_ylabel("Fuel", color=fuel_color, fontsize=8)
        ax_inv.tick_params(axis='y', labelcolor=fuel_color)
        ax_inv.set_ylim(0, capacity * 1.1)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

import os
if __name__ == "__main__":
    run_analysis()
