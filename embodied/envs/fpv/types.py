"""
All types use NED (North-East-Down) coordinate system:
  - +X = North (forward)
  - +Y = East (right)
  - +Z = Down (negative Z = altitude above ground)

Backends are responsible for converting their native coordinate
systems to/from NED internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class DroneState:
    """
    Drone state representation
    Attributes:
        position: [x, y, z] in meters (NED: +X=North, +Y=East, +Z=Down)
        velocity: [vx, vy, vz] in m/s (NED frame)
        orientation: [qw, qx, qy, qz] quaternion (scalar-first)
        angular_velocity: [wx, wy, wz] in rad/s (body frame)
        timestamp: Optional simulation timestamp
    """
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    timestamp: float = 0.0

    def __post_init__(self):
        """Ensure arrays are proper numpy arrays with correct shapes."""
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float32).reshape(3)
        self.orientation = np.asarray(self.orientation, dtype=np.float32).reshape(4)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float32).reshape(3)

    @property
    def altitude(self) -> float:
        """Altitude above ground (positive value, converts from NED Z)."""
        return -float(self.position[2])

    @property
    def speed(self) -> float:
        """Total speed magnitude in m/s."""
        return float(np.linalg.norm(self.velocity))

    def to_array(self) -> np.ndarray:
        """Convert to flat array [pos(3), vel(3), quat(4), ang_vel(3)] = 13 elements."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity,
        ]).astype(np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: float = 0.0) -> "DroneState":
        """Create from flat array."""
        arr = np.asarray(arr, dtype=np.float32)
        return cls(
            position=arr[0:3],
            velocity=arr[3:6],
            orientation=arr[6:10],
            angular_velocity=arr[10:13],
            timestamp=timestamp,
        )

    @classmethod
    def zeros(cls) -> "DroneState":
        """Create a zeroed state (hovering at origin)."""
        return cls(
            position=np.zeros(3, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Identity quaternion
            angular_velocity=np.zeros(3, dtype=np.float32),
        )


@dataclass
class ImageData:
    """Camera observation data.

    Attributes:
        rgb: RGB image as HxWx3 uint8 array
        depth: Optional depth image as HxWx1 uint8 array (255 = far/invalid)
        raw_rgb: Optional full-resolution RGB for video recording
    """
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    raw_rgb: Optional[np.ndarray] = None

    def __post_init__(self):
        """Ensure proper array types."""
        self.rgb = np.asarray(self.rgb, dtype=np.uint8)
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.uint8)
        if self.raw_rgb is not None:
            self.raw_rgb = np.asarray(self.raw_rgb, dtype=np.uint8)

    @property
    def size(self) -> Tuple[int, int]:
        """Image size as (height, width)."""
        return self.rgb.shape[:2]

    @classmethod
    def empty(cls, size: Tuple[int, int], include_depth: bool = False) -> "ImageData":
        """Create empty (black) image data."""
        h, w = size
        return cls(
            rgb=np.zeros((h, w, 3), dtype=np.uint8),
            depth=np.full((h, w, 1), 255, dtype=np.uint8) if include_depth else None,
        )


@dataclass
class ControlCommand:
    """Unified control command for rate-based control.

    All values are normalized to [-1, 1] (except throttle: [0, 1]).
    Backends handle conversion to their native control format.

    Attributes:
        throttle: Thrust command [0, 1], where ~0.5-0.6 is hover
        roll: Roll rate command [-1, 1], positive = roll right
        pitch: Pitch rate command [-1, 1], positive = pitch forward (nose down)
        yaw: Yaw rate command [-1, 1], positive = yaw right (clockwise from above)
    """
    throttle: float
    roll: float
    pitch: float
    yaw: float

    def __post_init__(self):
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.roll = float(self.roll)
        self.pitch = float(self.pitch)
        self.yaw = float(self.yaw)

    def to_array(self) -> np.ndarray:
        """Convert to [throttle, roll, pitch, yaw] array."""
        return np.array([self.throttle, self.roll, self.pitch, self.yaw], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ControlCommand":
        """Create from [throttle, roll, pitch, yaw] array."""
        arr = np.asarray(arr, dtype=np.float32).flatten()
        return cls(
            throttle=float(arr[0]),
            roll=float(arr[1]),
            pitch=float(arr[2]),
            yaw=float(arr[3]),
        )

    @classmethod
    def hover(cls) -> "ControlCommand":
        """Create a neutral hover command."""
        return cls(throttle=0.5, roll=0.0, pitch=0.0, yaw=0.0)


@dataclass
class MarkerConfig:
    """Configuration for visual waypoint markers."""
    target_asset: str = "BP_RedMarker"
    waypoint_asset: str = "BP_GrayMarker"
    spline_asset: str = "Sphere"
    target_scale: float = 0.8
    waypoint_scale: float = 0.5
    spline_scale: float = 0.12
    target_is_blueprint: bool = True
    waypoint_is_blueprint: bool = True
    spline_is_blueprint: bool = False


@dataclass
class RandomizationConfig:
    """Configuration for waypoint position randomization.

    Enables per-episode variation to prevent static path memorization.
    Uses momentum-based smoothing for natural flight paths.
    """
    enabled: bool = False
    xy_radius: float = 5.0
    z_range: Tuple[float, float] = (-2.0, 2.0)
    min_altitude: float = -5.0
    max_altitude: float = -12.0
    min_waypoint_distance: float = 8.0
    smoothing_factor: float = 0.6
    seed: Optional[int] = None


@dataclass
class WaypointConfig:
    """Configuration for the waypoint system."""
    positions: List[np.ndarray] = field(default_factory=list)
    reach_radius: float = 1.8
    loop_start_index: Optional[int] = 0
    visible_count: int = 5
    show_spline: bool = True
    spline_density: int = 20
    markers: MarkerConfig = field(default_factory=MarkerConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    def __post_init__(self):
        self.positions = [np.asarray(p, dtype=np.float32) for p in self.positions]


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    gate_radius: float = 1.8
    gate_bonus: float = 30.0
    gate_bonus_decay: bool = True
    gate_size: float = 1.8
    gate_bonus_progressive: bool = True
    gate_bonus_max_scale: float = 3.0
    collision_penalty: float = -25.0
    progress_scale: float = 5.0
    progress_clip_min: float = -0.4
    progress_clip_max: float = 0.8
    survival_bonus: float = 0.1
    survival_decay: float = 200.0
    time_penalty: float = -0.002
    time_penalty_growth: float = 300.0
    rate_penalty_scale: float = 350.0
    rate_penalty_control_hz: float = 100.0
    rate_yaw_weight: float = 0.3
    jerk_penalty_scale: float = 0.01
    visibility_penalty: float = 0.0
    camera_fov: float = 100.0


@dataclass
class ControlConfig:
    """Configuration for control scaling.

    The effective_range parameter accommodates tanh saturation in the policy.
    At effective_range=0.5, tanh gradient is 0.75 (healthy for learning).
    Action values beyond effective_range still work but clip at physical limits.
    """
    roll_scale: float = 8.0       # Base roll rate (rad/s) at effective_range
    pitch_scale: float = 8.0      # Base pitch rate (rad/s) at effective_range
    yaw_scale: float = 4.0        # Base yaw rate (rad/s) at effective_range
    throttle_hover: float = 0.35  # Physical hover throttle [0,1]
    effective_range: float = 0.5  # Action range for full physical output


@dataclass
class EnvConfig:
    """Main environment configuration."""
    size: Tuple[int, int] = (96, 96)
    control_hz: float = 100.0
    max_steps: int = 900
    use_depth: bool = False
