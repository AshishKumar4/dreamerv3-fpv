from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from embodied.envs.fpv.env import FPVEnv
from embodied.envs.fpv.backends.base import SimulatorBackend
from embodied.envs.fpv.rewards.base import RewardStrategy
from embodied.envs.fpv.waypoints.base import WaypointManager
from embodied.envs.fpv.types import (
    EnvConfig,
    ControlConfig,
    RewardConfig,
    WaypointConfig,
    MarkerConfig,
    RandomizationConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Registry
# =============================================================================

_BACKENDS: Dict[str, Type[SimulatorBackend]] = {}


def register_backend(name: str, backend_class: Type[SimulatorBackend]) -> None:
    _BACKENDS[name.lower()] = backend_class


def get_backend_class(name: str) -> Type[SimulatorBackend]:
    name = name.lower()
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return _BACKENDS[name]


# Register built-in backends (lazy import to avoid import errors)
def _register_builtin_backends():
    try:
        from embodied.envs.fpv.backends.colosseum import ColosseumBackend
        register_backend("colosseum", ColosseumBackend)
        register_backend("airsim", ColosseumBackend)  # Alias
    except ImportError:
        logger.debug("Colosseum backend not available (missing airsim)")


# =============================================================================
# Reward Registry
# =============================================================================

_REWARDS: Dict[str, Type[RewardStrategy]] = {}


def register_reward(name: str, reward_class: Type[RewardStrategy]) -> None:
    """Register a reward strategy implementation."""
    _REWARDS[name.lower()] = reward_class


def get_reward_class(name: str) -> Type[RewardStrategy]:
    """Get a registered reward class."""
    name = name.lower()
    if name not in _REWARDS:
        available = ", ".join(_REWARDS.keys())
        raise ValueError(f"Unknown reward '{name}'. Available: {available}")
    return _REWARDS[name]


def _register_builtin_rewards():
    """Register built-in reward strategies."""
    from embodied.envs.fpv.rewards.navigation import (
        NavigationReward,
    )
    register_reward("navigation", NavigationReward)
    register_reward("default", NavigationReward)


# =============================================================================
# Waypoint Manager Registry
# =============================================================================

_WAYPOINT_MANAGERS: Dict[str, Type[WaypointManager]] = {}


def register_waypoint_manager(name: str, manager_class: Type[WaypointManager]) -> None:
    """Register a waypoint manager implementation."""
    _WAYPOINT_MANAGERS[name.lower()] = manager_class


def get_waypoint_manager_class(name: str) -> Type[WaypointManager]:
    """Get a registered waypoint manager class."""
    name = name.lower()
    if name not in _WAYPOINT_MANAGERS:
        available = ", ".join(_WAYPOINT_MANAGERS.keys())
        raise ValueError(f"Unknown waypoint manager '{name}'. Available: {available}")
    return _WAYPOINT_MANAGERS[name]


def _register_builtin_waypoint_managers():
    """Register built-in waypoint managers."""
    from embodied.envs.fpv.waypoints.visual import VisualWaypointManager

    register_waypoint_manager("visual", VisualWaypointManager)
    register_waypoint_manager("default", VisualWaypointManager)


# =============================================================================
# Configuration Parsing
# =============================================================================

def parse_waypoints(raw: Union[List, None]) -> List[np.ndarray]:
    """Parse waypoint positions from config.

    Accepts either list of [x, y, z] or list of "x,y,z" strings.

    Args:
        raw: Raw waypoint list from config

    Returns:
        List of numpy arrays
    """
    if not raw:
        return []

    parsed = []
    for item in raw:
        if isinstance(item, str):
            try:
                parts = [float(x.strip()) for x in item.split(",")]
            except Exception:
                logger.warning(f"Failed to parse waypoint: {item}")
                continue
        else:
            parts = list(item)

        if len(parts) != 3:
            logger.warning(f"Waypoint must have 3 coordinates: {item}")
            continue

        parsed.append(np.array(parts, dtype=np.float32))

    return parsed


def build_env_config(config: Dict[str, Any]) -> EnvConfig:
    """Build EnvConfig from config dict."""
    defaults = EnvConfig()
    return EnvConfig(
        size=tuple(config.get("size", defaults.size)),
        control_hz=float(config.get("control_hz", defaults.control_hz)),
        max_steps=int(config.get("max_steps", defaults.max_steps)),
        use_depth=bool(config.get("use_depth", defaults.use_depth)),
    )


def build_control_config(config: Dict[str, Any]) -> ControlConfig:
    """Build ControlConfig from config dict."""
    defaults = ControlConfig()
    return ControlConfig(
        roll_scale=float(config.get("roll_scale", defaults.roll_scale)),
        pitch_scale=float(config.get("pitch_scale", defaults.pitch_scale)),
        yaw_scale=float(config.get("yaw_scale", defaults.yaw_scale)),
        throttle_hover=float(config.get("throttle_hover", defaults.throttle_hover)),
        effective_range=float(config.get("effective_range", defaults.effective_range)),
    )


def build_reward_config(config: Dict[str, Any]) -> RewardConfig:
    """Build RewardConfig from config dict."""
    defaults = RewardConfig()
    return RewardConfig(
        gate_radius=float(config.get("gate_radius", defaults.gate_radius)),
        gate_bonus=float(config.get("gate_bonus", defaults.gate_bonus)),
        gate_bonus_decay=bool(config.get("gate_bonus_decay", defaults.gate_bonus_decay)),
        gate_size=float(config.get("gate_size", defaults.gate_size)),
        gate_bonus_progressive=bool(config.get("gate_bonus_progressive", defaults.gate_bonus_progressive)),
        gate_bonus_max_scale=float(config.get("gate_bonus_max_scale", defaults.gate_bonus_max_scale)),
        collision_penalty=float(config.get("collision_penalty", defaults.collision_penalty)),
        progress_scale=float(config.get("progress_scale", defaults.progress_scale)),
        progress_clip_min=float(config.get("progress_clip_min", defaults.progress_clip_min)),
        progress_clip_max=float(config.get("progress_clip_max", defaults.progress_clip_max)),
        survival_bonus=float(config.get("survival_bonus", defaults.survival_bonus)),
        survival_decay=float(config.get("survival_decay", defaults.survival_decay)),
        time_penalty=float(config.get("time_penalty", defaults.time_penalty)),
        time_penalty_growth=float(config.get("time_penalty_growth", defaults.time_penalty_growth)),
        rate_penalty_scale=float(config.get("rate_penalty_scale", defaults.rate_penalty_scale)),
        rate_penalty_control_hz=float(config.get("rate_penalty_control_hz", defaults.rate_penalty_control_hz)),
        rate_yaw_weight=float(config.get("rate_yaw_weight", defaults.rate_yaw_weight)),
        jerk_penalty_scale=float(config.get("jerk_penalty_scale", defaults.jerk_penalty_scale)),
        visibility_penalty=float(config.get("visibility_penalty", defaults.visibility_penalty)),
        camera_fov=float(config.get("camera_fov", defaults.camera_fov)),
    )


def build_marker_config(config: Dict[str, Any]) -> MarkerConfig:
    """Build MarkerConfig from config dict."""
    defaults = MarkerConfig()
    return MarkerConfig(
        target_asset=config.get("target_asset", defaults.target_asset),
        waypoint_asset=config.get("waypoint_asset", defaults.waypoint_asset),
        spline_asset=config.get("spline_asset", defaults.spline_asset),
        target_scale=float(config.get("target_scale", defaults.target_scale)),
        waypoint_scale=float(config.get("waypoint_scale", defaults.waypoint_scale)),
        spline_scale=float(config.get("spline_scale", defaults.spline_scale)),
        target_is_blueprint=bool(config.get("target_is_blueprint", defaults.target_is_blueprint)),
        waypoint_is_blueprint=bool(config.get("waypoint_is_blueprint", defaults.waypoint_is_blueprint)),
        spline_is_blueprint=bool(config.get("spline_is_blueprint", defaults.spline_is_blueprint)),
    )


def build_randomization_config(config: Dict[str, Any]) -> RandomizationConfig:
    """Build RandomizationConfig from config dict.

    Args:
        config: Config dict, expects 'waypoint_randomization' key

    Returns:
        RandomizationConfig with parsed values
    """
    defaults = RandomizationConfig()
    rand_cfg = config.get("waypoint_randomization", {})

    # Parse z_range tuple
    z_range = rand_cfg.get("z_range", defaults.z_range)
    if isinstance(z_range, (list, tuple)) and len(z_range) == 2:
        z_range = (float(z_range[0]), float(z_range[1]))
    else:
        z_range = defaults.z_range

    return RandomizationConfig(
        enabled=bool(rand_cfg.get("enabled", defaults.enabled)),
        xy_radius=float(rand_cfg.get("xy_radius", defaults.xy_radius)),
        z_range=z_range,
        min_altitude=float(rand_cfg.get("min_altitude", defaults.min_altitude)),
        max_altitude=float(rand_cfg.get("max_altitude", defaults.max_altitude)),
        min_waypoint_distance=float(rand_cfg.get("min_waypoint_distance", defaults.min_waypoint_distance)),
        smoothing_factor=float(rand_cfg.get("smoothing_factor", defaults.smoothing_factor)),
        seed=rand_cfg.get("seed", defaults.seed),
    )


def build_waypoint_config(config: Dict[str, Any], marker_config: MarkerConfig) -> WaypointConfig:
    """Build WaypointConfig from config dict."""
    defaults = WaypointConfig()

    # Build randomization config
    randomization_config = build_randomization_config(config)

    return WaypointConfig(
        positions=parse_waypoints(config.get("static_gates", [])),
        reach_radius=float(config.get("gate_radius", defaults.reach_radius)),
        loop_start_index=config.get("loop_start_index", defaults.loop_start_index),
        visible_count=int(config.get("visible_waypoints", defaults.visible_count)),
        show_spline=bool(config.get("show_spline", defaults.show_spline)),
        spline_density=int(config.get("spline_samples_per_segment", defaults.spline_density)),
        markers=marker_config,
        randomization=randomization_config,
    )


# =============================================================================
# Factory Function
# =============================================================================

def create_fpv_env(
    simulator: str = "colosseum",
    reward_type: str = "navigation",
    waypoint_type: str = "visual",
    **config,
) -> FPVEnv:
    # Ensure registries are populated
    _register_builtin_backends()
    _register_builtin_rewards()
    _register_builtin_waypoint_managers()

    # Build configurations
    env_config = build_env_config(config)
    control_config = build_control_config(config)
    reward_config = build_reward_config(config)

    # Get simulator-specific marker config
    sim_config = config.get(simulator, {})
    markers_config = sim_config.get("markers", {})

    # Merge top-level marker settings (for backward compatibility)
    if "target_balloon_asset" in config:
        markers_config.setdefault("target_asset", config["target_balloon_asset"])
    if "waypoint_balloon_asset" in config:
        markers_config.setdefault("waypoint_asset", config["waypoint_balloon_asset"])
    if "target_is_blueprint" in config:
        markers_config.setdefault("target_is_blueprint", config["target_is_blueprint"])
    if "waypoint_is_blueprint" in config:
        markers_config.setdefault("waypoint_is_blueprint", config["waypoint_is_blueprint"])
    if "spline_is_blueprint" in config:
        markers_config.setdefault("spline_is_blueprint", config["spline_is_blueprint"])
    if "balloon_scale" in config:
        scale = config["balloon_scale"]
        if isinstance(scale, (list, tuple)):
            markers_config.setdefault("target_scale", scale[0])
            markers_config.setdefault("waypoint_scale", scale[0] * 0.6)

    marker_config = build_marker_config(markers_config)
    waypoint_config = build_waypoint_config(config, marker_config)

    # Create backend
    backend_class = get_backend_class(simulator)
    backend = _create_backend(backend_class, simulator, config)

    # Create reward strategy
    reward_class = get_reward_class(reward_type)
    reward = reward_class(reward_config)

    # Create waypoint manager
    waypoint_class = get_waypoint_manager_class(waypoint_type)
    waypoints = waypoint_class(waypoint_config)

    # Create and return environment
    return FPVEnv(
        backend=backend,
        reward_strategy=reward,
        waypoint_manager=waypoints,
        env_config=env_config,
        control_config=control_config,
    )


def _create_backend(
    backend_class: Type[SimulatorBackend],
    simulator: str,
    config: Dict[str, Any],
) -> SimulatorBackend:
    """Create and configure a backend instance."""
    sim_config = config.get(simulator, {})

    if simulator in ("colosseum", "airsim"):
        from embodied.envs.fpv.backends.colosseum import ColosseumConfig

        backend_config = ColosseumConfig(
            vehicle_name=sim_config.get("vehicle_name", ""),
        )
        return backend_class(backend_config)

    elif simulator in ("isaacsim", "isaac_sim", "pegasus"):
        from embodied.envs.fpv.backends.isaac_sim import IsaacSimConfig
        backend_config = IsaacSimConfig(
            host=sim_config.get("host", "localhost"),
            port=int(sim_config.get("port", 5555)),
            shm_name=sim_config.get("shm_name", "isaac_sim_fpv"),
            width=int(sim_config.get("camera_width", 640)),
            height=int(sim_config.get("camera_height", 480)),
            timeout=float(sim_config.get("timeout", 5.0)),
            max_retries=int(sim_config.get("max_retries", 3)),
        )
        return backend_class(backend_config)

    else:
        # Generic backend creation
        return backend_class()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_colosseum_env(**config) -> FPVEnv:
    """
    Create a Colosseum-based FPV environment.
    """
    return create_fpv_env(simulator="colosseum", **config)