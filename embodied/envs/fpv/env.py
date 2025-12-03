"""
Main simulation env class
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import elements
import embodied

from embodied.envs.fpv.backends.base import SimulatorBackend
from embodied.envs.fpv.rewards.base import RewardStrategy, RewardInfo
from embodied.envs.fpv.waypoints.base import WaypointManager
from embodied.envs.fpv.types import (
    DroneState,
    ControlCommand,
    ControlConfig,
    EnvConfig,
)

logger = logging.getLogger(__name__)

_ACTION_CHANNELS = ('throttle', 'roll', 'pitch', 'yaw')
_ACTION_LOG_KEYS = tuple(
    f"log/action_{ch}_{stat}"
    for stat in ('mean', 'std')
    for ch in _ACTION_CHANNELS
)
_ACTION_LOG_ZEROS = {k: np.float32(0) for k in _ACTION_LOG_KEYS}


class FPVEnv(embodied.Env):
    """FPV drone navigation environment.

    This environment provides a interface for training
    vision-based drone navigation across different simulators.

    The agent observes:
    - RGB camera image
    - Optional depth image

    The agent controls:
    - Throttle, roll rate, pitch rate, yaw rate

    The reward is based on:
    - Progress toward waypoints
    - Waypoint completion bonus
    - Collision penalty
    """

    def __init__(
        self,
        backend: SimulatorBackend,
        reward_strategy: RewardStrategy,
        waypoint_manager: WaypointManager,
        env_config: Optional[EnvConfig] = None,
        control_config: Optional[ControlConfig] = None,
    ):
        self._backend = backend
        self._reward = reward_strategy
        self._waypoints = waypoint_manager

        self._env_config = env_config or EnvConfig()
        self._control_config = control_config or ControlConfig()

        # Computed values
        self._dt = 1.0 / self._env_config.control_hz

        # Episode state
        self._steps: int = 0
        self._done: bool = True
        self._prev_dist: float = 0.0

        # Internal state for debugging (not exposed to agent)
        self._internal_state: Optional[DroneState] = None
        self._last_raw_frame: Optional[np.ndarray] = None

        # Track last command for telemetry/debugging
        self._last_command: ControlCommand = ControlCommand.hover()

        # Track last reward info for debugging/analysis
        self._last_reward_info: Optional[RewardInfo] = None
        self._last_waypoint_bonus: float = 0.0

        # Action logging for diagnostics
        self._episode_actions: List[np.ndarray] = []

    # =========================================================================
    # embodied.Env Interface
    # =========================================================================

    @property
    def obs_space(self) -> Dict[str, Any]:
        """Observation space specification.

        Returns dict of Space objects for embodied compatibility.

        Note: 'state' and 'goal' are privileged observations for auxiliary
        loss training only. They are NOT fed to the policy - the policy
        only sees the image.
        """
        size = self._env_config.size
        spaces = {
            "image": elements.Space(np.uint8, size + (3,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "state": elements.Space(np.float32, (13,)),
            "goal": elements.Space(np.float32, (4,)),
            **{k: elements.Space(np.float32) for k in _ACTION_LOG_KEYS},
        }

        if self._env_config.use_depth:
            spaces["depth"] = elements.Space(np.uint8, size + (1,))

        return spaces

    @property
    def act_space(self) -> Dict[str, Any]:
        return {
            "reset": elements.Space(bool),
            "action": elements.Space(np.float32, (4,), -1.0, 1.0),
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        reset = bool(action.get("reset", False))
        if reset or self._done:
            return self._reset()

        # Parse and scale action
        ctrl = np.clip(np.asarray(action["action"], dtype=np.float32), -1.0, 1.0)
        self._episode_actions.append(ctrl.copy())

        # Convert action to control command
        command = self._action_to_command(ctrl)
        self._last_command = command  # Store for external access

        # Step backend
        state, has_collided = self._backend.step(command, self._dt)
        self._internal_state = state

        # Check waypoint reached
        # Capture target position BEFORE advance for offset calculation
        target_before_advance = self._waypoints.get_current_target()
        logger.debug(f"[ENV] check_and_advance: pos={state.position}, target={target_before_advance}")
        reached_index = self._waypoints.check_and_advance(state.position)
        waypoint_bonus = 0.0

        if reached_index is not None:
            logger.info(f"[ENV] WAYPOINT REACHED! reached_index={reached_index}")
            # Waypoint reached - compute offset from waypoint center
            # (used for SkyDreamer-style spatial gate bonus decay)
            offset = None
            if target_before_advance is not None:
                offset = state.position - target_before_advance

            waypoint_bonus = self._reward.on_waypoint_reached(
                reached_index, self._waypoints.num_waypoints, offset
            )

            # Update visuals
            self._waypoints.update_visuals(self._backend, reached_index)

            # Update reward strategy with new target distance
            new_target = self._waypoints.get_current_target()
            if new_target is not None:
                new_dist = float(np.linalg.norm(new_target - state.position))
                self._reward.update_target(new_dist)

        # Compute reward
        target = self._waypoints.get_current_target()
        if target is None:
            target = state.position  # No target, use current position

        reward_info = self._reward.compute(
            state=state,
            target_position=target,
            step=self._steps,
            max_steps=self._env_config.max_steps,
            has_collided=has_collided,
        )

        # Store for debugging/telemetry access
        self._last_reward_info = reward_info
        self._last_waypoint_bonus = waypoint_bonus

        # Add waypoint bonus
        total_reward = reward_info.total + waypoint_bonus

        # Update step counter
        self._steps += 1
        timeout = self._steps >= self._env_config.max_steps

        is_terminal = reward_info.is_terminal
        is_last = is_terminal or timeout
        self._done = is_last

        # Get observation
        obs = self._get_observation()
        obs.update(
            reward=np.float32(total_reward),
            is_first=False,
            is_last=is_last,
            is_terminal=is_terminal,
        )

        obs.update(self._compute_privileged_obs(state, target))
        obs.update(self._compute_action_stats(is_last))

        return obs

    def render(self) -> Optional[np.ndarray]:
        return self._last_raw_frame

    def close(self) -> None:
        try:
            self._waypoints.cleanup_visuals(self._backend)
            self._backend.disconnect()
        except Exception as e:
            logger.warning(f"Error during close: {e}")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _reset(self) -> Dict[str, Any]:
        # Log previous episode stats
        if self._episode_actions:
            actions = np.array(self._episode_actions)
            mean = actions.mean(axis=0)
            std = actions.std(axis=0)
            logger.info(
                f"[Episode] Actions - mean: [{mean[0]:.2f}, {mean[1]:.2f}, "
                f"{mean[2]:.2f}, {mean[3]:.2f}] std: [{std[0]:.2f}, {std[1]:.2f}, "
                f"{std[2]:.2f}, {std[3]:.2f}] steps: {len(actions)}"
            )

        # Reset state
        self._done = False
        self._steps = 0
        self._episode_actions = []

        # Reset backend
        state = self._backend.reset()
        self._internal_state = state

        # Reset waypoints
        self._waypoints.reset()

        # Initialize visuals
        self._waypoints.initialize_visuals(self._backend)

        # Initialize reward tracking
        target = self._waypoints.get_current_target()
        if target is not None:
            initial_dist = float(np.linalg.norm(target - state.position))
        else:
            initial_dist = 0.0

        self._reward.reset(initial_dist)
        self._last_command = ControlCommand.hover()  # Reset to neutral

        # Get initial observation
        obs = self._get_observation()
        obs.update(
            reward=np.float32(0.0),
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

        obs.update(self._compute_privileged_obs(state, target))
        obs.update(self._compute_action_stats(is_last=False))

        return obs

    def _compute_action_stats(self, is_last: bool) -> Dict[str, np.ndarray]:
        """Compute action statistics for logging"""
        if not (is_last and self._episode_actions):
            return _ACTION_LOG_ZEROS
        actions = np.array(self._episode_actions)
        mean, std = actions.mean(axis=0), actions.std(axis=0)
        stats = np.concatenate([mean, std])
        return {k: np.float32(stats[i]) for i, k in enumerate(_ACTION_LOG_KEYS)}

    def _compute_privileged_obs(
        self,
        state: "DroneState",
        target: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute privileged state/goal observations for auxiliary loss.

        These are SkyDreamer-style training targets that teach the encoder
        to extract physics-relevant features.

        Args:
            state: Current drone state (position, velocity, orientation, omega)
            target: Current waypoint target position (NED), or None

        Returns:
            Dict with 'state' (13-dim) and 'goal' (4-dim) arrays
        """
        # State: [pos(3), vel(3), quat(4), omega(3)] = 13 dims
        state_vec = np.concatenate([
            state.position,
            state.velocity,
            state.orientation,
            state.angular_velocity,
        ]).astype(np.float32)

        # Goal: [direction(3), distance(1)] = 4 dims
        if target is not None:
            to_target = target - state.position
            distance = float(np.linalg.norm(to_target))
            if distance > 1e-6:
                direction = to_target / distance
            else:
                direction = np.zeros(3, dtype=np.float32)
            goal_vec = np.concatenate([
                direction.astype(np.float32),
                np.array([distance], dtype=np.float32),
            ])
        else:
            goal_vec = np.zeros(4, dtype=np.float32)

        return {
            "state": state_vec,
            "goal": goal_vec,
        }

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from backend."""
        size = self._env_config.size
        use_depth = self._env_config.use_depth

        images = self._backend.get_images(size, include_depth=use_depth)
        self._last_raw_frame = images.raw_rgb

        obs = {
            "image": images.rgb.astype(np.uint8),
        }

        if use_depth and images.depth is not None:
            obs["depth"] = images.depth

        return obs

    def _action_to_command(self, action: np.ndarray) -> ControlCommand:
        """Convert raw action to control command with scaling.

        Args:
            action: Raw action [throttle, roll, pitch, yaw] in [-1, 1]

        Returns:
            Scaled ControlCommand
        """
        cfg = self._control_config

        tanh_scale = 1.0 / cfg.effective_range  # 2.0 for range 0.5

        # Throttle: asymmetric mapping centered at hover
        # Action 0 = hover, Action ±effective_range = [0, 1]
        hover = cfg.throttle_hover
        scaled_throttle = float(action[0]) * tanh_scale
        if scaled_throttle >= 0:
            throttle = hover + scaled_throttle * (1.0 - hover)
        else:
            throttle = hover + scaled_throttle * hover
        throttle = np.clip(throttle, 0.0, 1.0)

        # Rate commands: symmetric
        roll = float(action[1]) * tanh_scale * cfg.roll_scale
        pitch = float(action[2]) * tanh_scale * cfg.pitch_scale
        yaw = float(action[3]) * tanh_scale * cfg.yaw_scale

        return ControlCommand(
            throttle=throttle,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def backend(self) -> SimulatorBackend:
        """Access the simulator backend."""
        return self._backend

    @property
    def waypoint_manager(self) -> WaypointManager:
        """Access the waypoint manager."""
        return self._waypoints

    @property
    def reward_strategy(self) -> RewardStrategy:
        """Access the reward strategy."""
        return self._reward

    @property
    def internal_state(self) -> Optional[DroneState]:
        """Get last known drone state (for debugging, not exposed to agent)."""
        return self._internal_state

    @property
    def reward_info(self) -> Optional[RewardInfo]:
        return self._last_reward_info

    @property
    def waypoint_bonus(self) -> float:
        return self._last_waypoint_bonus

    def get_info(self) -> Dict[str, Any]:
        """Get environment information for logging."""
        return {
            "backend": self._backend.get_info(),
            "steps": self._steps,
            "current_waypoint": self._waypoints.current_index,
            "num_waypoints": self._waypoints.num_waypoints,
        }
