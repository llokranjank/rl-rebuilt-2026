from dataclasses import dataclass
from typing import List, Optional
import config

class Zone:
    def __init__(self, name: str, initial_balls: int):
        self.name = name
        self.balls = initial_balls

    def add_balls(self, count: int):
        self.balls += count

    def remove_balls(self, count: int) -> int:
        """Removes up to 'count' balls, returns actual removed."""
        removed = min(self.balls, count)
        self.balls -= removed
        return removed

    def __repr__(self):
        return f"Zone({self.name}, balls={self.balls})"

class Robot:
    def __init__(self, team: str, id: int, capacity: int, transit_time: float, 
                 balls_shot_per_sec: float, intake_efficiency: float, start_zone: str, 
                 initial_inventory: int = 0):
        self.team = team # 'Red' or 'Blue'
        self.id = id
        self.capacity = capacity
        self.transit_time = transit_time
        self.balls_shot_per_sec = balls_shot_per_sec
        self.intake_efficiency = max(0.0, min(1.0, intake_efficiency))
        
        # State
        self.start_zone = start_zone
        self.current_zone_name = start_zone
        self.inventory = initial_inventory
        self.action_cooldown = 0.0 # Time until robot is free
        self.current_action = "Idle"
        
        # Stats
        self.score_contribution = 0

    def is_busy(self) -> bool:
        return self.action_cooldown > 0

    def update(self, dt: float):
        if self.action_cooldown > 0:
            self.action_cooldown -= dt
            if self.action_cooldown <= 0:
                self.action_cooldown = 0
                self.current_action = "Idle"

    def set_action_transit(self, target_zone: str):
        self.current_action = f"Moving to {target_zone}"
        self.action_cooldown = self.transit_time
        # Location update happens in Engine upon completion.

    def set_action_shoot(self):
        self.current_action = "Shooting"
        # Cooldown is set by Engine based on dynamic rate

    def set_action_pass(self):
        self.current_action = "Passing"
        # Cooldown is set by Engine based on dynamic rate
        
    def __repr__(self):
        return f"{self.team}Bot{self.id}(loc={self.current_zone_name}, inv={self.inventory}/{self.capacity})"
