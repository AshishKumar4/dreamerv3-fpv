#!/usr/bin/env python3
"""
Manual Control Testing Script for DreamerV3 FPV Environment.
"""

import time
import csv
import signal
import argparse
import threading
import logging
import numpy as np
import pygame
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Dict, Any
from scipy.spatial.transform import Rotation
from ruamel import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.getLogger('embodied.envs.fpv.waypoints').setLevel(logging.DEBUG)
logging.getLogger('embodied.envs.fpv.env').setLevel(logging.DEBUG)

from embodied.envs.fpv.factory import create_fpv_env
from embodied.run.video_utils import get_raw_frame
from embodied.core.replay import Replay

CONFIGS_PATH = Path(__file__).parent / "dreamerv3" / "configs.yaml"


def load_fpv_config() -> Dict[str, Any]:
    with open(CONFIGS_PATH) as f:
        configs = yaml.YAML(typ='safe').load(f)

    fpv_config = configs.get('defaults', {}).get('env', {}).get('fpv', {})

    if not fpv_config:
        raise ValueError(f"No fpv config found in {CONFIGS_PATH}")

    print(f"[Config] Loaded FPV config from {CONFIGS_PATH}")
    return fpv_config


# =============================================================================
# Remote Controller
# =============================================================================

class RemoteController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick found! Please connect TX16s.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"[Remote] Connected: {self.joystick.get_name()}")
        print(f"[Remote] Axes: {self.joystick.get_numaxes()}, Buttons: {self.joystick.get_numbuttons()}")

        # Configuration
        self.deadzone = 0.05
        self.roll_axis = 0      # Left stick X
        self.pitch_axis = 1     # Left stick Y
        self.throttle_axis = 2  # Right stick Y (or slider)
        self.yaw_axis = 3       # Right stick X

    def read_controls(self) -> np.ndarray:
        pygame.event.pump()

        # Read raw axes (all in -1..1)
        roll_raw = self.joystick.get_axis(self.roll_axis)
        pitch_raw = self.joystick.get_axis(self.pitch_axis)
        throttle_raw = self.joystick.get_axis(self.throttle_axis)
        yaw_raw = self.joystick.get_axis(self.yaw_axis)

        # Apply deadzone
        roll = self._apply_deadzone(roll_raw)
        pitch = self._apply_deadzone(pitch_raw)
        yaw = -self._apply_deadzone(yaw_raw)  # Reversed for TX16s

        # Throttle: Keep in [-1, 1] (env will map to hover-centered)
        throttle = throttle_raw

        return np.array([throttle, roll, pitch, yaw], dtype=np.float32)

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def cleanup(self):
        if self.joystick:
            self.joystick.quit()
        pygame.quit()


# =============================================================================
# Display Manager 
# =============================================================================

class DisplayManager(threading.Thread):
    def __init__(self, image_size=(480, 640), use_depth=False):
        super().__init__(daemon=True, name="DisplayThread")

        self.image_height, self.image_width = image_size
        self.use_depth = use_depth
        self.running = False

        self.display_queue = Queue(maxsize=2)

        self.screen = None
        self.font_small = None
        self.font_medium = None

        # FPS tracking
        self.frame_times = []
        self.last_fps_update = time.time()
        self.current_fps = 0.0

        # Exit flag
        self.exit_requested = False

    def run(self):
        # Initialize pygame in this thread
        pygame.init()

        if self.use_depth:
            total_width = self.image_width + self.image_width + 300  # FPV + Depth + Telemetry
        else:
            total_width = self.image_width + 300  # FPV + Telemetry only
        total_height = self.image_height

        self.screen = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("FPV Manual Control - DreamerV3")

        # Fonts
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 22)

        # Layout regions
        self.fpv_rect = pygame.Rect(0, 0, self.image_width, self.image_height)
        if self.use_depth:
            self.depth_rect = pygame.Rect(self.image_width, 0, self.image_width, self.image_height)
            self.telem_rect = pygame.Rect(self.image_width * 2, 0, 300, self.image_height)
        else:
            self.depth_rect = None
            self.telem_rect = pygame.Rect(self.image_width, 0, 300, self.image_height)

        print(f"[Display] Window created: {total_width}x{total_height}")
        print(f"[Display] Thread running at ~30 Hz")

        self.running = True

        while self.running:
            frame_start = time.time()

            try:
                data = self.display_queue.get(timeout=0.1)

                # Render frame
                self._render_frame(
                    data['fpv_img'],
                    data['depth_img'],
                    data['telemetry']
                )

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit_requested = True
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            self.exit_requested = True
                            self.running = False

                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)

                if time.time() - self.last_fps_update >= 1.0:
                    if self.frame_times:
                        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                        self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    self.frame_times.clear()
                    self.last_fps_update = time.time()

                time.sleep(1.0 / 30.0)

            except Empty:
                continue
            except Exception as e:
                print(f"[Display] Error: {e}")
                break

        pygame.quit()
        print("[Display] Thread stopped")

    def update(self, fpv_img, depth_img, telemetry):
        if not self.running:
            return

        try:
            if self.display_queue.full():
                try:
                    self.display_queue.get_nowait()
                except Empty:
                    pass

            self.display_queue.put_nowait({
                'fpv_img': fpv_img,
                'depth_img': depth_img,
                'telemetry': telemetry
            })
        except:
            pass

    def _render_frame(self, fpv_img, depth_img, telemetry):

        self.screen.fill((0, 0, 0))

        if fpv_img is not None:
            fpv_surface = self._numpy_to_surface(fpv_img)
            self.screen.blit(fpv_surface, self.fpv_rect)

        if depth_img is not None and self.depth_rect is not None:
            depth_rgb = np.repeat(depth_img, 3, axis=2)
            depth_surface = self._numpy_to_surface(depth_rgb)
            self.screen.blit(depth_surface, self.depth_rect)

        telem_surface = self._create_telemetry_surface(telemetry)
        self.screen.blit(telem_surface, self.telem_rect)

        pygame.display.flip()

    def _numpy_to_surface(self, img):
        return pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))

    def _create_telemetry_surface(self, data):
        surface = pygame.Surface((300, self.image_height))
        surface.fill((0, 0, 0))

        y = 10
        line_height = 20

        # Title
        title = self.font_medium.render("TELEMETRY", True, (0, 255, 255))
        surface.blit(title, (10, y))
        y += line_height * 2

        # Build text lines
        texts = [
            ("Orientation:", (0, 255, 0)),
            (f"  w: {data.get('qw', 0):.3f}", (0, 255, 0)),
            (f"  x: {data.get('qx', 0):.3f}", (0, 255, 0)),
            (f"  y: {data.get('qy', 0):.3f}", (0, 255, 0)),
            (f"  z: {data.get('qz', 0):.3f}", (0, 255, 0)),
            ("", (0, 0, 0)),  # Spacer
            ("Velocity (m/s):", (0, 255, 0)),
            (f"  vx: {data.get('vx', 0):.2f}", (0, 255, 0)),
            (f"  vy: {data.get('vy', 0):.2f}", (0, 255, 0)),
            (f"  vz: {data.get('vz', 0):.2f}", (0, 255, 0)),
            ("", (0, 0, 0)),
            ("Angular (rad/s):", (0, 255, 0)),
            (f"  wx: {data.get('wx', 0):.2f}", (0, 255, 0)),
            (f"  wy: {data.get('wy', 0):.2f}", (0, 255, 0)),
            (f"  wz: {data.get('wz', 0):.2f}", (0, 255, 0)),
            ("", (0, 0, 0)),
            ("Commands:", (255, 255, 0)),
            (f"  Thr: {data.get('cmd_throttle', 0):.2f}", (255, 255, 0)),
            (f"  Rol: {data.get('cmd_roll', 0):.2f}", (255, 255, 0)),
            (f"  Pit: {data.get('cmd_pitch', 0):.2f}", (255, 255, 0)),
            (f"  Yaw: {data.get('cmd_yaw', 0):.2f}", (255, 255, 0)),
            ("", (0, 0, 0)),
            ("Reward Breakdown:", (255, 165, 0)),
            (f"  Total: {data.get('reward_total', 0):.3f}", (255, 165, 0)),
            (f"  Progress: {data.get('reward_progress', 0):.3f}", (100, 255, 100)),
            (f"  Survival: {data.get('reward_survival', 0):.3f}", (100, 255, 100)),
            (f"  Rate: {data.get('reward_rate_penalty', 0):.3f}", (255, 100, 100)),
            (f"  Jerk: {data.get('reward_jerk_penalty', 0):.3f}", (255, 100, 100)),
            (f"  Time: {data.get('reward_time_penalty', 0):.3f}", (255, 100, 100)),
            (f"  Visibility: {data.get('reward_visibility_penalty', 0):.3f}", (255, 100, 100)),
            (f"  WP Bonus: {data.get('reward_waypoint_bonus', 0):.1f}", (255, 255, 0)),
            ("", (0, 0, 0)),
            ("Performance:", (255, 255, 0)),
            (f"  Display: {data.get('display_fps', 0):.1f} FPS", (255, 255, 0)),
            (f"  Control: {data.get('control_hz', 0):.1f} Hz", (255, 255, 0)),
        ]

        if 'gate_distance' in data:
            texts.extend([
                ("", (0, 0, 0)),
                (f"Gate {data.get('gate_index', 0)}/{data.get('gate_total', 0)}", (255, 0, 255)),
                (f"Dist: {data.get('gate_distance', 0):.1f}m", (255, 0, 255)),
            ])

        for text, color in texts:
            rendered = self.font_small.render(text, True, color)
            surface.blit(rendered, (10, y))
            y += line_height

        return surface

    def get_fps(self) -> float:
        """Get current display FPS."""
        return self.current_fps

    def stop(self):
        """Stop display thread."""
        self.running = False


# =============================================================================
# Telemetry Logger
# =============================================================================

class TelemetryLogger:

    def __init__(self, log_dir="flight_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"flight_{timestamp}.csv"

        self.file = open(self.log_file, 'w', newline='')
        self.writer = csv.writer(self.file)

        # Write header
        self.writer.writerow([
            'timestamp', 'step',
            'qw', 'qx', 'qy', 'qz',
            'vx', 'vy', 'vz',
            'wx', 'wy', 'wz',
            'action_throttle', 'action_roll', 'action_pitch', 'action_yaw',
            'cmd_throttle', 'cmd_roll', 'cmd_pitch', 'cmd_yaw',
            # Reward components
            'reward_total', 'reward_progress', 'reward_survival',
            'reward_rate_penalty', 'reward_jerk_penalty', 'reward_time_penalty',
            'reward_visibility_penalty', 'reward_waypoint_bonus',
            'gate_distance', 'gate_index'
        ])

        print(f"[Logger] Logging to: {self.log_file}")

    def log_state(self, step, data):
        self.writer.writerow([
            time.time(), step,
            data.get('qw', 0), data.get('qx', 0), data.get('qy', 0), data.get('qz', 0),
            data.get('vx', 0), data.get('vy', 0), data.get('vz', 0),
            data.get('wx', 0), data.get('wy', 0), data.get('wz', 0),
            data.get('action_throttle', 0), data.get('action_roll', 0),
            data.get('action_pitch', 0), data.get('action_yaw', 0),
            data.get('cmd_throttle', 0), data.get('cmd_roll', 0),
            data.get('cmd_pitch', 0), data.get('cmd_yaw', 0),
            # Reward components
            data.get('reward_total', 0), data.get('reward_progress', 0),
            data.get('reward_survival', 0), data.get('reward_rate_penalty', 0),
            data.get('reward_jerk_penalty', 0), data.get('reward_time_penalty', 0),
            data.get('reward_visibility_penalty', 0), data.get('reward_waypoint_bonus', 0),
            data.get('gate_distance', 0), data.get('gate_index', 0)
        ])
        self.file.flush()

    def close(self):
        self.file.close()
        print(f"[Logger] Log closed")


# =============================================================================
# Main Control System
# =============================================================================

class FPVManualControlSystem:
    def __init__(self, args):
        print("\n" + "="*70)
        print("FPV MANUAL CONTROL - Pure FPVEnv Interface")
        print("="*70)

        # Load config from dreamerv3/configs.yaml
        print("\n[Env] Creating FPVEnv...")
        fpv_config = load_fpv_config()

        # Override with command-line args
        if args.size:
            fpv_config['size'] = args.size
        if args.with_depth:
            fpv_config['use_depth'] = args.with_depth
        if args.gates:
            fpv_config['static_gates'] = args.gates

        # Create environment using config from configs.yaml
        self.env = create_fpv_env(**fpv_config)

        print(f"[Env] Size: {self.env._env_config.size}")
        print(f"[Env] Control: {self.env._env_config.control_hz} Hz")
        print(f"[Env] Max steps: {self.env._env_config.max_steps}")

        # Components
        self.remote = RemoteController()
        self.display = DisplayManager(
            image_size=(480, 640),  # Native camera resolution, not observation size
            use_depth=self.env._env_config.use_depth
        )
        self.logger = TelemetryLogger() if args.log else None

        # Demo collection
        self.replay = None
        if args.collect:
            collect_dir = Path(args.collect_dir)
            collect_dir.mkdir(parents=True, exist_ok=True)
            self.replay = Replay(
                length=64,
                capacity=None,
                directory=collect_dir / "chunks",
                chunksize=1024,
                online=False,
            )
            print(f"[Collect] Demo collection enabled -> {collect_dir}/chunks/")

        # State
        self.running = False
        self.obs = None
        self.step_count = 0

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)

        print("\nPress Ctrl+C or 'q' in display window to exit")
        print("="*70 + "\n")

    def _signal_handler(self, sig, frame):
        print("\n[System] Shutdown requested...")
        self.running = False

    def _make_transition(self, obs, action):
        return {
            'image': obs['image'],
            'action': action,
            'reward': np.float32(obs['reward']),
            'is_first': bool(obs['is_first']),
            'is_last': bool(obs['is_last']),
            'is_terminal': bool(obs.get('is_terminal', False)),
            'state': obs['state'],
            'goal': obs['goal'],
        }

    def run(self):
        # Start display thread
        self.display.start()
        time.sleep(0.5)  # Let display initialize

        # Initialize environment
        print("[Control] Resetting environment...")
        self.obs = self.env.step({"reset": True})
        self.running = True
        self.step_count = 0

        # Collect reset observation (with zero action, as no action was taken yet)
        if self.replay is not None:
            zero_action = np.zeros(4, dtype=np.float32)
            self.replay.add(self._make_transition(self.obs, zero_action), worker=0)
            self.step_count += 1

        test_frame = get_raw_frame(self.env)
        if test_frame is not None:
            print(f"[Control] Raw frame check: OK - shape {test_frame.shape} (native camera resolution)")
        else:
            print("[Control] WARNING: get_raw_frame() returned None! Videos will be blurry 96x96")

        loop_period = 1.0 / 100.0  # 10 ms

        print("[Control] Starting 100 Hz control loop...")
        print("[Control] Fly the drone with TX16s!\n")

        # Performance tracking
        loop_times = []
        control_loop_times = []
        last_stats_update = time.time()
        current_control_hz = 100.0  # Initialize to target
        current_display_fps = 30.0  # Initialize to target

        while self.running:
            loop_start = time.time()

            t_control = time.time()
            action = self.remote.read_controls()

            t_env = time.time()
            self.obs = self.env.step({"action": action})
            env_step_time = time.time() - t_env
            self.step_count += 1

            if env_step_time > 0.050:  # More than 50ms
                print(f"[Warning] Slow env.step(): {env_step_time*1000:.1f}ms")

            # Demo collection
            if self.replay is not None:
                self.replay.add(self._make_transition(self.obs, action), worker=0)

            fpv_image = get_raw_frame(self.env) 
            if fpv_image is None:
                print("[Warning] get_raw_frame() returned None, using resized obs")
                fpv_image = self.obs["image"]  # Fallback to 96x96 resized
            depth_image = self.obs.get("depth")  # Optional depth
            reward = float(self.obs["reward"])
            is_last = bool(self.obs["is_last"])
            state = self.env.internal_state
            gate_info = self._get_gate_info()
            telemetry = {
                # State
                'qw': state.orientation[0],
                'qx': state.orientation[1],
                'qy': state.orientation[2],
                'qz': state.orientation[3],
                'vx': state.velocity[0],
                'vy': state.velocity[1],
                'vz': state.velocity[2],
                'wx': state.angular_velocity[0],
                'wy': state.angular_velocity[1],
                'wz': state.angular_velocity[2],

                # Actions (as sent to env)
                'action_throttle': float(action[0]),
                'action_roll': float(action[1]),
                'action_pitch': float(action[2]),
                'action_yaw': float(action[3]),

                # Actual commands (after scaling)
                'cmd_throttle': self.env._last_command.throttle,
                'cmd_roll': self.env._last_command.roll,
                'cmd_pitch': self.env._last_command.pitch,
                'cmd_yaw': self.env._last_command.yaw,

                # Reward breakdown
                'reward_total': reward,
                'reward_progress': self.env.reward_info.progress if self.env.reward_info else 0.0,
                'reward_survival': self.env.reward_info.survival if self.env.reward_info else 0.0,
                'reward_rate_penalty': self.env.reward_info.rate_penalty if self.env.reward_info else 0.0,
                'reward_jerk_penalty': self.env.reward_info.jerk_penalty if self.env.reward_info else 0.0,
                'reward_time_penalty': self.env.reward_info.time_penalty if self.env.reward_info else 0.0,
                'reward_visibility_penalty': self.env.reward_info.visibility_penalty if self.env.reward_info else 0.0,
                'reward_waypoint_bonus': self.env.waypoint_bonus,

                # Performance metrics
                'display_fps': current_display_fps,
                'control_hz': current_control_hz,

                # Gate tracking
                **gate_info,
            }

            self.display.update(fpv_image, depth_image, telemetry)

            if self.logger:
                self.logger.log_state(self.step_count, telemetry)

            target = self.env.waypoint_manager.get_current_target()
            if target is not None:
                dist = np.linalg.norm(target - state.position)
                if dist < 5.0:  # Log when within 5m
                    print(f"[DEBUG] Dist to target: {dist:.2f}m | Target: {target} | Pos: {state.position} | (need < 1.8m)")

            if is_last:
                # Determine termination reason
                is_terminal = bool(self.obs.get("is_terminal", False))
                reason = "COLLISION" if is_terminal else "TIMEOUT/MAX_STEPS"
                print(f"\n[Control] Episode ended: {reason} at step {self.step_count}")
                self.running = False

            if self.display.exit_requested:
                print(f"\n[Control] Exit requested from display")
                self.running = False

            elapsed = time.time() - loop_start
            sleep_time = loop_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

            loop_times.append(elapsed)
            control_loop_times.append(elapsed)
            if len(loop_times) > 100:
                loop_times.pop(0)

            if time.time() - last_stats_update >= 1.0:
                if control_loop_times:
                    avg_control_period = sum(control_loop_times) / len(control_loop_times)
                    current_control_hz = 1.0 / avg_control_period if avg_control_period > 0 else 0

                current_display_fps = self.display.get_fps()

                control_loop_times.clear()
                last_stats_update = time.time()

            if elapsed > loop_period * 1.5 and self.step_count % 100 == 0:
                avg_time = sum(loop_times) / len(loop_times)
                print(f"[Warning] Control loop slow: {elapsed*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")

        # Cleanup
        print("\n[System] Shutting down...")
        self.display.stop()
        self.display.join(timeout=2.0)  # Wait for display thread to stop
        if self.logger:
            self.logger.close()
        if self.replay is not None:
            self.replay.save()
            print(f"[Collect] Saved {self.step_count} steps to replay buffer")
        try:
            self.env.close()
        except Exception as e:
            print(f"[System] Error during env.close(): {e}")
        try:
            self.remote.cleanup()
        except Exception as e:
            print(f"[System] Error during remote.cleanup(): {e}")

        # Print stats
        if loop_times:
            avg_time = sum(loop_times) / len(loop_times)
            max_time = max(loop_times)
            print(f"\n[Stats] Control loop timing:")
            print(f"  Average: {avg_time*1000:.2f}ms")
            print(f"  Max: {max_time*1000:.2f}ms")
            print(f"  Target: {loop_period*1000:.2f}ms (100 Hz)")
            print(f"  Total steps: {self.step_count}")

        print("\n[System] Shutdown complete")

    def _get_gate_info(self) -> Dict[str, Any]:
        state = self.env.internal_state
        target = self.env.waypoint_manager.get_current_target()

        if target is None:
            return {}

        # Distance
        distance = np.linalg.norm(target - state.position)

        # Relative position in body frame
        rel_world = target - state.position

        # Transform to body frame using quaternion
        quat_scipy = [
            state.orientation[1],  # x
            state.orientation[2],  # y
            state.orientation[3],  # z
            state.orientation[0],  # w (scipy format)
        ]
        r = Rotation.from_quat(quat_scipy)
        rel_body = r.inv().apply(rel_world)

        return {
            'gate_x': float(rel_body[0]),
            'gate_y': float(rel_body[1]),
            'gate_z': float(rel_body[2]),
            'gate_distance': float(distance),
            'gate_index': self.env.waypoint_manager.current_index + 1,
            'gate_total': self.env.waypoint_manager.num_waypoints,
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Manual control testing for DreamerV3 FPV environment"
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=[96, 96],
        help="Image size (height width), default: 96 96 (matches Dreamer FPV config)"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable CSV logging"
    )
    parser.add_argument(
        "--with-depth",
        action="store_true",
        help="Enable depth image (Dreamer FPV config uses depth)"
    )
    parser.add_argument(
        "--gates",
        nargs="+",
        help="Custom gate positions (e.g., '8,0,-3' '18,0,-5')"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Enable demo collection for world model pretraining"
    )
    parser.add_argument(
        "--collect-dir",
        default="demo_data",
        help="Directory to save demo data (default: demo_data)"
    )

    args = parser.parse_args()

    try:
        system = FPVManualControlSystem(args)
        system.run()
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
