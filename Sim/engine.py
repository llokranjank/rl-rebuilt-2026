import config
from entities import Zone, Robot
from typing import List, Dict

class SimulationEngine:
    def __init__(self):
        self.time = 0.0
        self.zones: Dict[str, Zone] = {
            config.ZONE_RED: Zone(config.ZONE_RED, 0),
            config.ZONE_BLUE: Zone(config.ZONE_BLUE, 0),
            config.ZONE_NEUTRAL: Zone(config.ZONE_NEUTRAL, config.TOTAL_BALLS) 
        }
        self.robots: List[Robot] = []
        
        # Game State
        self.red_hub_status = config.HUB_ACTIVE
        self.blue_hub_status = config.HUB_ACTIVE
        self.red_score = 0.0 # Score can be float during calc? No, integer.
        self.blue_score = 0.0
        
        # Analytics
        self.history_time = []
        self.history_red_balls = []
        self.history_blue_balls = []
        self.history_neutral_balls = []
        self.history_robot_states = [] # List of (Time, RobotIndex, Zone, Action)
        self.history_hub_status = [] # List of (Time, RedStatus, BlueStatus)
        self.history_score = [] # List of (Time, RedScore, BlueScore)
        self.history_zone_balls = [] # List of (Time, RedCount, NeutralCount, BlueCount)

    def add_robot(self, robot: Robot):
        self.robots.append(robot)

    def get_zone_density(self, zone_name: str) -> float:
        if zone_name not in self.zones: return 0.0
        # Formula uses "BallsInCurrentZone / 504"
        return self.zones[zone_name].balls / config.TOTAL_BALLS

    def calculate_intake_rate(self, robot: Robot) -> float:
        density = self.get_zone_density(robot.current_zone_name)
        # Rate = Efficiency * Density * 10
        return robot.intake_efficiency * density * config.MAX_INTAKE_RATE

    def step(self, dt: float):
        self.time += dt
        
        # 1. Update Game Rules
        self._update_hub_status()
        
        # 2. Update Robots
        for bot in self.robots:
            prev_cooldown = bot.action_cooldown
            # Capture action BEFORE update because update() might reset it to 'Idle'
            # Actually, let's check if Robot.update resets it.
            # Yes it does. So we need the action name.
            action_being_performed = bot.current_action
            
            # Continuous Logic: Intake
            if bot.current_action == "Intaking" and bot.action_cooldown > 0:
                self._process_intake(bot, dt)
            
            # Cooldown Tick
            bot.update(dt)
            
            # Action Completion Trigger (Falling Edge of cooldown)
            if prev_cooldown > 0 and bot.action_cooldown <= 0:
                self._on_action_complete(bot, action_being_performed)

        # 3. Snapshot
        self.history_time.append(self.time)
        self.history_red_balls.append(self.zones[config.ZONE_RED].balls)
        self.history_blue_balls.append(self.zones[config.ZONE_BLUE].balls)
        self.history_neutral_balls.append(self.zones[config.ZONE_NEUTRAL].balls)
        
        for idx, bot in enumerate(self.robots):
            self.history_robot_states.append({
                'time': self.time,
                'robot_idx': idx,
                'team': bot.team,
                'zone': bot.current_zone_name,
                'action': bot.current_action,
                'inventory': bot.inventory,
                'cooldown': bot.action_cooldown # For animation progress
            })
            
        self.history_hub_status.append({
            'time': self.time,
            'red': self.red_hub_status,
            'blue': self.blue_hub_status
        })

        self.history_score.append({
            'time': self.time,
            'red': self.red_score,
            'blue': self.blue_score
        })

        self.history_zone_balls.append({
            'time': self.time,
            'red': self.zones[config.ZONE_RED].balls,
            'neutral': self.zones[config.ZONE_NEUTRAL].balls,
            'blue': self.zones[config.ZONE_BLUE].balls
        })

    def _process_intake(self, bot: Robot, dt: float):
        # Calculate max balls validation
        space = bot.capacity - bot.inventory
        if space <= 0: return

        rate = self.calculate_intake_rate(bot)
        amount = rate * dt
        
        # Clamp to available balls in zone and robot capacity
        zone = self.zones[bot.current_zone_name]
        possible_from_zone = zone.balls
        actual_amount = min(amount, space, possible_from_zone)
        
        # Execute transfer
        # Note: Handling float accumulation? 
        # Simulating discrete balls with floats is tricky. 
        # Let's keep floats for rates, but floor for "useful" balls? 
        # Or just track float inventory and display int? 
        # User said "Global Ball Count: 504 (Integers)". 
        # We should probably accumulate 'partial' balls and only transfer when >= 1?
        # For simplicity in this v21 engine, let's treat balls as Floats internally 
        # to ensure smooth rates, but rounding for display.
        # Or: Accumulate 'uncollected' balance.
        
        # Let's simple-transfer floats.
        zone.balls -= actual_amount
        bot.inventory += actual_amount

    def _on_action_complete(self, bot: Robot, action_name: str):
        action = action_name
        
        if action.startswith("Moving"):
            # Format: "Moving to Red"
            target = action.split(" to ")[1]
            bot.current_zone_name = target # Arrived!
            
        elif action == "Shooting":
            # Transfer balls Robot -> Hub -> Neutral
            count = bot.inventory
            bot.inventory = 0
            
            # Scoring
            # Points count only if Alliance Hub is Active
            points = 0
            if bot.team == "Red" and self.red_hub_status == config.HUB_ACTIVE:
                points = count
                self.red_score += points
            elif bot.team == "Blue" and self.blue_hub_status == config.HUB_ACTIVE:
                points = count
                self.blue_score += points
            
            # Recirculate to Neutral
            self.zones[config.ZONE_NEUTRAL].balls += count
            
        elif action == "Passing":
            # Transfer balls Robot -> Alliance Zone
            count = bot.inventory
            bot.inventory = 0
            
            target_zone = config.ZONE_RED if bot.team == "Red" else config.ZONE_BLUE
            self.zones[target_zone].balls += count
            
        bot.current_action = "Idle"

    def _update_hub_status(self):
        # Endgame: Last 30s
        if config.MATCH_DURATION - self.time <= 30:
            self.red_hub_status = config.HUB_ACTIVE
            self.blue_hub_status = config.HUB_ACTIVE
            return

        if self.time <= config.AUTO_DURATION:
            self.red_hub_status = config.HUB_ACTIVE
            self.blue_hub_status = config.HUB_ACTIVE
        else:
            teleop_time = self.time - config.SHIFT_START_TIME
            if teleop_time < 0:
                pass # Gap logic, assume unchanged or active
            else:
                shift_index = int(teleop_time // config.SHIFT_DURATION)
                if shift_index % 2 == 0:
                    self.red_hub_status = config.HUB_ACTIVE
                    self.blue_hub_status = config.HUB_INACTIVE
                else:
                    self.red_hub_status = config.HUB_INACTIVE
                    self.blue_hub_status = config.HUB_ACTIVE

    # --- Commands ---
    def command_move(self, robot_idx: int, target_zone: str):
        bot = self.robots[robot_idx]
        if bot.is_busy(): return
        
        # Rule: Must travel via Neutral Zone
        # Red <-> Blue is invalid
        current = bot.current_zone_name
        
        # If trying to cross directly between Alliance Zones, forbid it
        if (current == config.ZONE_RED and target_zone == config.ZONE_BLUE) or \
           (current == config.ZONE_BLUE and target_zone == config.ZONE_RED):
            return

        bot.set_action_transit(target_zone)

    def command_intake(self, robot_idx: int, duration: float):
        bot = self.robots[robot_idx]
        if bot.is_busy(): return
        bot.current_action = "Intaking"
        bot.action_cooldown = duration

    def command_shoot(self, robot_idx: int):
        bot = self.robots[robot_idx]
        if bot.is_busy(): return

        # Rule: Can only shoot from Alliance Zone
        if bot.current_zone_name != bot.start_zone:
            return 
        
        # Dynamic Duration
        # Duration = Inventory / BallsShotPerSec
        # Avoid div by zero
        rate = bot.balls_shot_per_sec if bot.balls_shot_per_sec > 0 else 1.0
        duration = bot.inventory / rate
        if duration < 0.1: duration = 0.1 # Min duration for state visibility
        
        bot.action_cooldown = duration
        bot.set_action_shoot()

    def command_pass(self, robot_idx: int):
        bot = self.robots[robot_idx]
        if bot.is_busy(): return
        
        # Rule: Can only Pass from Neutral or Opponent Zone (i.e., NOT Alliance Zone)
        if bot.current_zone_name == bot.start_zone:
            return

        # Dynamic Duration (Pass uses same rate as shoot?)
        # User said "Shooting Time: Seconds required to empty the magazine (used for both Shoot and Pass)."
        # So yes, use same rate.
        rate = bot.balls_shot_per_sec if bot.balls_shot_per_sec > 0 else 1.0
        duration = bot.inventory / rate
        if duration < 0.1: duration = 0.1 

        bot.action_cooldown = duration
        bot.set_action_pass()
