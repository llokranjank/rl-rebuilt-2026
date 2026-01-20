import gymnasium as gym
from gymnasium import spaces
import numpy as np
from . import config
from .engine import SimulationEngine
from .entities import Robot
from stable_baselines3 import PPO
import os

class LogisticsEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.engine = None
        
        # Action Space: MultiDiscrete([6, 6, 6])
        # Controls Red Robot 1, Red Robot 2, Red Robot 3
        # Actions: 
        # 0: Move Alliance
        # 1: Move Opponent
        # 2: Move Neutral
        # 3: Intake, 4: Shoot, 5: Pass
        self.action_space = spaces.MultiDiscrete([6, 6, 6])

        # Observation Space:
        # For each of 6 robots: [Loc(3), Inv(1), Action(5)] = 9 features -> 54 total
        # Global: MyHub(1), OpHub(1), ZoneDensities(3), Time(1) -> 6 total
        # Total: 60
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(60,), dtype=np.float32)
        
        # Opponent Policy (Self-Play)
        self.opponent_model = None
        self.opponent_path = "ppo_logistics_opponent.zip"
        self._load_opponent()

    def _load_opponent(self):
        if os.path.exists(self.opponent_path):
            try:
                # Load model without environment to check its params
                model = PPO.load(self.opponent_path)
                
                # Verify Observation Shape
                # PPO model stores observation_space. If it doesn't match ours, discard.
                # Note: stable_baselines3 models might have 'observation_space' attribute
                if model.observation_space is not None:
                     if model.observation_space.shape != self.observation_space.shape:
                         print(f"Opponent Model shape mismatch ({model.observation_space.shape} vs {self.observation_space.shape}). Discarding.")
                         self.opponent_model = None
                         return

                self.opponent_model = model
                print(f"Loaded Opponent Model from {self.opponent_path}")
            except Exception as e:
                print(f"Failed to load Opponent Model ({e}). Blue will be idle/random.")
                self.opponent_model = None
        else:
            print("No Opponent Model found (First run). Blue will be idle/random.")
            self.opponent_model = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize Engine
        self.engine = SimulationEngine()
        
        # Add Robots: 3 Red, 3 Blue
        # Red Team (Indices 0, 1, 2 = IDs 1, 2, 3)
        for i in range(1, 4):
            # Index 0 is Robot 1 (i=1)
            cfg = config.ROBOT_CONFIGS[i-1].copy()
            if 'id' in cfg: del cfg['id']
            if 'type' in cfg: del cfg['type']
            
            red_bot = Robot(team="Red", id=i, start_zone=config.ZONE_RED, initial_inventory=8, **cfg)
            self.engine.add_robot(red_bot)
            
        # Blue Team (Indices 3, 4, 5 = IDs 1, 2, 3)
        for i in range(1, 4):
            cfg = config.ROBOT_CONFIGS[i-1].copy()
            if 'id' in cfg: del cfg['id']
            if 'type' in cfg: del cfg['type']
            
            blue_bot = Robot(team="Blue", id=i, start_zone=config.ZONE_BLUE, initial_inventory=8, **cfg)
            self.engine.add_robot(blue_bot)
        
        total_preloaded = 48
        self.engine.zones[config.ZONE_NEUTRAL].balls -= total_preloaded
        
        self.last_score = 0
        
        # Reload opponent if file changed (simulating iterative training)
        # In a real heavy loop we might not want to reload every reset, but for this scale it's fine.
        if options and options.get("reload_opponent"):
            self._load_opponent()
        
        return self._get_obs(team="Red"), {}

    def step(self, action):
        # 1. Get Blue Actions (Opponent)
        blue_actions = [0, 0, 0] # Default Idle/invalid
        
        if self.opponent_model:
            # Generate Blue View (Flipped)
            blue_obs = self._get_obs(team="Blue")
            blue_actions, _ = self.opponent_model.predict(blue_obs, deterministic=True)
        else:
            # Fallback: Random or Simple Static
            # Let's just do random valid moves to avoid being totally dead
            blue_actions = self.action_space.sample()

        # 2. Apply Red Actions (Indices 0, 1, 2)
        for i, a in enumerate(action):
            self._apply_relative_action(bot_idx=i, action=a, team="Red")
            
        # 3. Apply Blue Actions (Indices 3, 4, 5)
        for i, a in enumerate(blue_actions):
            self._apply_relative_action(bot_idx=i+3, action=a, team="Blue")

        # 4. Step Engine
        SIM_STEP = 0.1
        DECISION_INTERVAL = 1.0
        
        for _ in range(int(DECISION_INTERVAL / SIM_STEP)):
            self.engine.step(SIM_STEP)

        # 5. Calculate Reward (Team Reward for Red)
        current_score = self.engine.red_score
        reward = current_score - self.last_score
        self.last_score = current_score
        
        # Terminated
        terminated = self.engine.time >= config.MATCH_DURATION
        truncated = False
        
        if terminated:
            margin = self.engine.red_score - self.engine.blue_score
            if margin > 0:
                reward += 100.0 
                reward += margin
        
        # reward -= 0.01 # Time penalty removed

        obs = self._get_obs(team="Red")
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _apply_relative_action(self, bot_idx, action, team):
        # 0: Move Alliance, 1: Move Opponent, 2: Move Neutral
        # 3: Intake, 4: Shoot, 5: Pass
        
        alliance_zone = config.ZONE_RED if team == "Red" else config.ZONE_BLUE
        opponent_zone = config.ZONE_BLUE if team == "Red" else config.ZONE_RED
        
        if action == 0: self.engine.command_move(bot_idx, alliance_zone)
        elif action == 1: self.engine.command_move(bot_idx, opponent_zone)
        elif action == 2: self.engine.command_move(bot_idx, config.ZONE_NEUTRAL)
        elif action == 3: self.engine.command_intake(bot_idx, duration=1.0)
        elif action == 4: self.engine.command_shoot(bot_idx)
        elif action == 5: self.engine.command_pass(bot_idx)

    def _get_obs(self, team="Red"):
        # We need to render the state from the perspective of 'team'.
        # If team is Blue, we swap Red/Blue data so the agent always "feels" like Red.
        
        obs_list = []
        
        # Determine "My" and "Opponent" mappings
        my_zone = config.ZONE_RED if team == "Red" else config.ZONE_BLUE
        op_zone = config.ZONE_BLUE if team == "Red" else config.ZONE_RED
        
        # 1. Robots: Need to list MY robots first, then OPPONENT robots
        # Engine always stores [R1, R2, R3, B1, B2, B3]
        
        if team == "Red":
            my_bots = self.engine.robots[0:3]
            op_bots = self.engine.robots[3:6]
        else:
            my_bots = self.engine.robots[3:6]
            op_bots = self.engine.robots[0:3]
            
        all_bots_ordered = my_bots + op_bots
        
        for bot in all_bots_ordered:
            # Loc One-Hot (3): [MyZone, OpZone, Neutral]
            loc = [0, 0, 0]
            if bot.current_zone_name == my_zone: loc[0] = 1
            elif bot.current_zone_name == op_zone: loc[1] = 1
            elif bot.current_zone_name == config.ZONE_NEUTRAL: loc[2] = 1
            
            # Inventory (1)
            inv = bot.inventory / bot.capacity

            # Action One-Hot (5): [Idle, Moving, Intaking, Shooting, Passing]
            act = [0, 0, 0, 0, 0]
            a_str = bot.current_action
            if a_str == "Idle": act[0] = 1
            elif a_str.startswith("Moving"): act[1] = 1
            elif a_str == "Intaking": act[2] = 1
            elif a_str == "Shooting": act[3] = 1
            elif a_str == "Passing": act[4] = 1
            else: act[0] = 1 # Fallback to Idle
            
            obs_list.extend(loc)
            obs_list.append(inv)
            obs_list.extend(act)
            
        # 2. Global State
        # Hub Active (1) - Is MY Hub Active?
        if team == "Red":
            my_hub_active = 1.0 if self.engine.red_hub_status == config.HUB_ACTIVE else 0.0
            op_hub_active = 1.0 if self.engine.blue_hub_status == config.HUB_ACTIVE else 0.0
        else:
            my_hub_active = 1.0 if self.engine.blue_hub_status == config.HUB_ACTIVE else 0.0
            op_hub_active = 1.0 if self.engine.red_hub_status == config.HUB_ACTIVE else 0.0
            
        # Densities (3) - [MyZone, OpZone, Neutral]
        den_my = self.engine.get_zone_density(my_zone)
        den_op = self.engine.get_zone_density(op_zone)
        den_n = self.engine.get_zone_density(config.ZONE_NEUTRAL)
        
        # Time (1)
        time_norm = self.engine.time / config.MATCH_DURATION
        
        obs_list.extend([my_hub_active, op_hub_active, den_my, den_op, den_n, time_norm])
        
        return np.array(obs_list, dtype=np.float32)
