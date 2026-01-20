import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sim import config
from sim.gym_env import LogisticsEnv
from stable_baselines3 import PPO

# --- Visual Constraints ---
FIELD_WIDTH = 100
FIELD_HEIGHT = 50
ZONE_WIDTH = FIELD_WIDTH / 3

ZONE_COORDS = {
    config.ZONE_RED: FIELD_WIDTH * 0.16,     # Center of Red Zone
    config.ZONE_NEUTRAL: FIELD_WIDTH * 0.5,  # Center of Neutral
    config.ZONE_BLUE: FIELD_WIDTH * 0.84     # Center of Blue Zone
}

TEAM_COLOR = {'Red': '#FF4444', 'Blue': '#4444FF'}
HUB_COLOR = {'ACTIVE': 'gold', 'INACTIVE': 'gray'}

def run_and_animate(model_path="ppo_logistics_v21_3v3", save_path="match_simulation.gif", headless=True):
    # 1. Run Simulation to collect data
    print(f"Running Match for Animation (Model: {model_path})...")
    env = LogisticsEnv()
    
    # Try to load model
    model = None
    if model_path and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print("Loaded model for visualization.")
    else:
        print("No model found/provided. Using random actions.")
        
    obs, _ = env.reset(options={"reload_opponent": True})
    engine = env.engine
    
    # Collect history
    done = False
    time_limit = 0
    while not done and time_limit < 2000:
        if model:
            try:
                action, _ = model.predict(obs, deterministic=True)
            except ValueError as e:
                print(f"Model prediction failed (likely shape mismatch): {e}")
                print("Falling back to random actions for remainder of match.")
                model = None
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()
            
        obs, reward, done, truncated, info = env.step(action)
        time_limit += 1
    
    print(f"Match Finished. Score: R {engine.red_score} - B {engine.blue_score}")
    print("Generating Animation (this may take a minute)...")
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.set_xlim(0, FIELD_WIDTH)
    ax.set_ylim(0, FIELD_HEIGHT)
    ax.set_aspect('equal')
    ax.axis('off') # Hide axes
    
    # Background Zones
    # Red Zone
    ax.add_patch(patches.Rectangle((0, 0), ZONE_WIDTH, FIELD_HEIGHT, color='#FFEEEE', zorder=0))
    ax.text(ZONE_WIDTH/2, 2, "RED ZONE", ha='center', color='pink', fontweight='bold')
    
    # Neutral Zone
    ax.add_patch(patches.Rectangle((ZONE_WIDTH, 0), ZONE_WIDTH, FIELD_HEIGHT, color='#F0F0F0', zorder=0))
    ax.text(FIELD_WIDTH/2, 2, "NEUTRAL ZONE", ha='center', color='gray', fontweight='bold')
    
    # Blue Zone
    ax.add_patch(patches.Rectangle((ZONE_WIDTH*2, 0), ZONE_WIDTH, FIELD_HEIGHT, color='#EEEEFF', zorder=0))
    ax.text(FIELD_WIDTH*0.84, 2, "BLUE ZONE", ha='center', color='lightblue', fontweight='bold')
    
    # Hubs (Circles at ends)
    red_hub = patches.Circle((5, FIELD_HEIGHT/2), 3, color='gray', zorder=1)
    blue_hub = patches.Circle((FIELD_WIDTH-5, FIELD_HEIGHT/2), 3, color='gray', zorder=1)
    ax.add_patch(red_hub)
    ax.add_patch(blue_hub)
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='NONE', edgecolor='black', label='Actions:'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='Shoot'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Intake'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Pass'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, label='Move'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=5, frameon=False)
    
    # Scoreboard
    score_text = ax.text(FIELD_WIDTH/2, FIELD_HEIGHT-3, "0 - 0", ha='center', fontsize=16, fontweight='bold', zorder=5)
    time_text = ax.text(FIELD_WIDTH/2, FIELD_HEIGHT-6, "T: 0s", ha='center', fontsize=10, zorder=5)
    
    # Robots (6 total)
    robot_patches = []
    robot_texts = []
    robot_status_dots = []
    
    y_positions = np.linspace(10, FIELD_HEIGHT-10, 6)
    
    for i in range(6):
        bot = engine.robots[i]
        c = TEAM_COLOR[bot.team]
        
        # Main Body
        circ = patches.Circle((0, 0), 2.5, color=c, zorder=3)
        ax.add_patch(circ)
        robot_patches.append(circ)
        
        # Inventory Text
        txt = ax.text(0, 0, "0", ha='center', va='center', color='white', fontsize=9, fontweight='bold', zorder=4)
        robot_texts.append(txt)
        
        # Action Dot
        dot = patches.Circle((0, 0), 1.0, color='white', zorder=4)
        ax.add_patch(dot)
        robot_status_dots.append(dot)

    # Data Pre-processing
    times = engine.history_time
    n_steps = len(times)
    
    def get_robot_state(step_idx, bot_idx):
        flat_idx = step_idx * 6 + bot_idx
        if flat_idx >= len(engine.history_robot_states):
             return engine.history_robot_states[-1] 
        return engine.history_robot_states[flat_idx]
    
    def update(frame):
        current_time = times[frame]
        
        # Update Score
        s_red = engine.history_score[frame]['red']
        s_blue = engine.history_score[frame]['blue']
        score_text.set_text(f"{int(s_red)} - {int(s_blue)}")
        time_text.set_text(f"T: {int(current_time)}s")
        
        # Update Hubs
        h_red_status = engine.history_hub_status[frame]['red']
        h_blue_status = engine.history_hub_status[frame]['blue']
        red_hub.set_color(HUB_COLOR[h_red_status])
        blue_hub.set_color(HUB_COLOR[h_blue_status])
        
        # Update Robots
        for i in range(6):
            state = get_robot_state(frame, i)
            
            # Position Smoothing
            target_x = ZONE_COORDS[state['zone']]
            x_pos = target_x
            
            if "Moving to" in state['action']:
                target_name = state['action'].replace("Moving to ", "")
                dest_x = ZONE_COORDS.get(target_name, target_x)
                
                cfg_idx = i % 3
                TRANSIT_TOTAL = config.ROBOT_CONFIGS[cfg_idx]['transit_time']
                
                progress = 1.0 - (state['cooldown'] / TRANSIT_TOTAL)
                progress = max(0.0, min(1.0, progress))
                
                x_pos = target_x + (dest_x - target_x) * progress
            
            y_pos = y_positions[i]
            
            robot_patches[i].set_center((x_pos, y_pos))
            robot_texts[i].set_position((x_pos, y_pos))
            robot_texts[i].set_text(f"{int(state['inventory'])}")
            
            # Action Dot
            act = state['action']
            dot_color = 'none' 
            if "Shooting" in act: dot_color = 'yellow'
            elif "Intaking" in act: dot_color = 'green'
            elif "Passing" in act: dot_color = 'orange'
            elif "Moving" in act: dot_color = 'white'
            
            robot_status_dots[i].set_center((x_pos + 2, y_pos + 2))
            robot_status_dots[i].set_color(dot_color)
    
    # Create Animation
    skip = 2
    frames = range(0, n_steps, skip)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    # Save
    print(f"Saving to {save_path}...")
    try:
        ani.save(save_path, writer='pillow', fps=20)
        print(f"Done! Saved {save_path}")
    except Exception as e:
        print(f"Failed to save GIF to {save_path}: {e}")
    finally:
        plt.close()

import os
if __name__ == "__main__":
    run_and_animate()
