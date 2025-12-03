"""Abstract base class for reward strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from embodied.envs.fpv.types import DroneState, RewardConfig


@dataclass
class RewardInfo:
    total: float
    progress: float = 0.0
    survival: float = 0.0
    waypoint_bonus: float = 0.0
    time_penalty: float = 0.0
    collision_penalty: float = 0.0
    rate_penalty: float = 0.0
    jerk_penalty: float = 0.0
    visibility_penalty: float = 0.0
    is_terminal: bool = False
    distance_to_target: float = 0.0


class RewardStrategy(ABC):
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._prev_distance: float = 0.0
        self._prev_omega: Optional[np.ndarray] = None  # For jerk penalty

    def reset(self, initial_distance: float) -> None:
        self._prev_distance = initial_distance
        self._prev_omega = None  # Reset jerk penalty state

    @abstractmethod
    def compute(
        self,
        state: DroneState,
        target_position: np.ndarray,
        step: int,
        max_steps: int,
        has_collided: bool,
    ) -> RewardInfo:
        pass

    @abstractmethod
    def on_waypoint_reached(
        self,
        waypoint_index: int,
        num_waypoints: int,
        offset: Optional[np.ndarray] = None,
    ) -> float:
        pass

    def update_target(self, new_distance: float) -> None:
        self._prev_distance = new_distance

    @property
    def reach_radius(self) -> float:
        return self.config.gate_radius
