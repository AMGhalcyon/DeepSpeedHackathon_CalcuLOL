#!/usr/bin/env python3
"""The driver's control path, with no ROS in it.

`driver.py` used to hold both the ROS wiring and the algorithm. That meant the
offline harness had to keep its own copy of the control path, and the two
drifted: `lap_sim.py` fed the MPC its previous *MPC* steer while `driver.py`
fed it the *gap-follower's* steer, so the harness was not testing the thing the
judge runs. Anything measured against that copy was measuring a different car.

So the algorithm lives here and the node is wiring. There is one control path
and both the node and the harness drive it.

Nothing in this module imports rclpy. Logging goes through a duck-typed object
with `info`/`warn`/`error`; the node passes `self.get_logger()` and the harness
passes whatever it likes (or nothing, for silence).
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
try:
    from .mpc_controller import MPCController   # ROS / package import
    from .vehicle_model import VehicleModel
except ImportError:
    from mpc_controller import MPCController    # standalone `python3 lap_sim.py`
    from vehicle_model import VehicleModel


class _NullLogger:
    """Swallows log calls, so the core runs with no logger attached."""

    def info(self, _msg):
        pass

    def warn(self, _msg):
        pass

    def error(self, _msg):
        pass


@dataclass(frozen=True)
class DriverConfig:
    """Everything the control path is allowed to be tuned by.

    Defaults mirror `driver.py`'s `declare_parameter` calls exactly. Keeping
    them equal means a forgotten `--driver-params` flag is a provable no-op
    rather than a silently different car.
    """

    crawl_speed: float = 1.0        # [m/s]
    max_range: float = 8.0          # [m] clip the scan here
    v_max: float = 3.0              # [m/s] speed cap on the profile
    lat_accel: float = 0.8          # [m/s^2] cornering budget
    a_brake: float = 0.8            # [m/s^2] braking budget
    steer_limit: float = 0.4        # [rad]


class DriveCommand(NamedTuple):
    """One command destined for /drive.

    `steering_angle_velocity` is carried but not yet populated -- the judged
    config runs the simulator in STEERING_ANGLE mode and ignores it. It is here
    so the output-conditioning work has somewhere to put it.
    """

    steering_angle: float
    speed: float
    steering_angle_velocity: float = 0.0


DT_CLAMP_LOW = 0.5     # x nominal: a burst of scans must not zero the interval
DT_CLAMP_HIGH = 3.0    # x nominal: a dropped scan must not hand over a 1 s dt


class DtTracker:
    """The interval the plant advanced between sensor stamps, clamped.

    Stamps come from the bridge's own sim clock, so consecutive deltas are the
    real hold period regardless of wall-clock load or real-time factor.
    """

    def __init__(self, nominal):
        self.nominal = nominal
        self.dt = nominal
        self._last_stamp = None
        self._history = []

    def update(self, stamp):
        if stamp is None:
            return
        if self._last_stamp is not None:
            self.dt = float(np.clip(stamp - self._last_stamp,
                                    DT_CLAMP_LOW * self.nominal,
                                    DT_CLAMP_HIGH * self.nominal))
            self._history.append(self.dt)
        self._last_stamp = stamp

    def summary(self):
        """(mean, p95, max) of the dt seen since the last reset."""
        if not self._history:
            return (self.nominal,) * 3
        h = np.asarray(self._history)
        return float(h.mean()), float(np.percentile(h, 95)), float(h.max())

    def reset_stats(self):
        self._history = []


class DriverCore:
    """The algorithm: reference path, MPC, and the reactive safety layer."""

    def __init__(self,
                 cfg: DriverConfig = None,
                 mpc_controller: MPCController = None,
                 logger=None):
        self.cfg = cfg if cfg is not None else DriverConfig()
        self.log = logger if logger is not None else _NullLogger()

        # Unpacked onto self so the moved code reads as it did in the node.
        self.crawl_speed = self.cfg.crawl_speed
        self.max_range = self.cfg.max_range
        self.v_max = self.cfg.v_max
        self.lat_accel = self.cfg.lat_accel
        self.a_brake = self.cfg.a_brake
        self.steer_limit = self.cfg.steer_limit

        self.mpc_controller = (mpc_controller if mpc_controller is not None
                               else MPCController(vehicle_model=VehicleModel()))
        self.mpc_dt = self.mpc_controller.dt
        self._dt = DtTracker(self.mpc_dt)
        self._planned_scan_t = None
        self._last_planned = DriveCommand(0.0, 0.0, 0.0)

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]
        # Sideslip and yaw rate arrive free on /ego_racecar/odom. Recorded now,
        # not yet consumed: the MPC still assumes beta = 0.
        self.beta = 0.0           # [rad]
        self.yaw_rate = 0.0       # [rad/s]
        self.t_odom = None        # [s] stamp the pose was measured at

        # Latest scan, as the three fields preprocess() needs.
        self._scan_ranges = None
        self._scan_angle_min = 0.0
        self._scan_angle_inc = 0.0
        self.t_scan = None        # [s] stamp the scan was measured at

        self.reference_path = None
        self.reference_speeds = None

        # What actually went on the wire last, and when.
        self._last_cmd = DriveCommand(0.0, 0.0, 0.0)
        self._last_publish_t = None

        self._load_reference_path()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def set_odom(self, x, y, yaw, v, beta=0.0, yaw_rate=0.0, t_sim=None):
        """Latest pose and velocity, already unpacked from whatever carried it."""
        self.position = (x, y)
        self.yaw = yaw
        self.speed = v
        self.beta = beta
        self.yaw_rate = yaw_rate
        self.t_odom = t_sim

    def set_scan(self, ranges, angle_min, angle_inc, t_sim=None):
        """Latest LiDAR scan, as the raw range array plus its angular frame."""
        self._scan_ranges = ranges
        self._scan_angle_min = angle_min
        self._scan_angle_inc = angle_inc
        if t_sim != self.t_scan:
            self._dt.update(t_sim)
        self.t_scan = t_sim

    @property
    def dt_measured(self):
        """[s] Hold interval between the last two distinct scan stamps."""
        return self._dt.dt

    def dt_summary(self):
        """(mean, p95, max) of dt_measured; clears the window."""
        summary = self._dt.summary()
        self._dt.reset_stats()
        return summary

    # ------------------------------------------------------------------
    # Reference path loading
    # ------------------------------------------------------------------
    def _centerline_candidates(self):
        """Places icra26_centerline.csv can legitimately live.

        The package is installed under race_ws/install/... but the map ships at
        the repo root, so the obvious 'share directory' guess does not exist.
        Try the ROS way first, then walk up from this file to the repo root.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = []

        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('team_driver'), 'maps'))
        except Exception:
            pass

        # Walk up looking for a maps/ dir holding the file. Seven levels covers
        # install/team_driver/lib/pythonX.Y/site-packages/team_driver/.
        node = here
        for _ in range(8):
            node = os.path.dirname(node)
            candidates.append(os.path.join(node, 'maps'))

        return [os.path.join(c, 'icra26_centerline.csv') for c in candidates]

    def _load_reference_path(self):
        """Load centerline as reference path."""
        try:
            centerline_path = next(
                (p for p in self._centerline_candidates() if os.path.exists(p)), None)

            if centerline_path is None:
                self.log.error(
                    'icra26_centerline.csv not found in any of: '
                    f'{self._centerline_candidates()}. MPC will NOT be used; '
                    'falling back to the reactive gap-following controller.')
                self.reference_path = None
                return

            # The file is commented CSV; np.loadtxt needs comments='#' and the
            # header lines dropped.
            centerline_data = np.loadtxt(centerline_path, delimiter=',', comments='#')
            centerline_data = np.atleast_2d(centerline_data)

            self.log.info(
                f'Loaded {len(centerline_data)} centerline points from {centerline_path}')

            self.reference_path = centerline_data[:, :2]

            # Curvature-limited speeds, not a flat 3 m/s. The track is ~0.5 m
            # wide and the centerline peaks well above a curvature that 3 m/s
            # can hold, so a constant reference drives straight into the
            # outside wall at the first real corner.
            self.reference_speeds = self.mpc_controller.speed_profile(
                v_max=self.v_max, lat_accel=self.lat_accel, a_brake=self.a_brake,
                path=self.reference_path)
            self.log.info(
                f'Speed profile: {self.reference_speeds.min():.2f}'
                f'-{self.reference_speeds.max():.2f} m/s')

            self.mpc_controller.set_reference_path(self.reference_path, self.reference_speeds)

        except Exception as e:
            self.log.error(f'Error loading reference path: {e}')
            self.reference_path = None

    # ------------------------------------------------------------------
    # One control cycle
    # ------------------------------------------------------------------
    def step(self, t_sim=None) -> DriveCommand:
        """Plan one command from the latest sensor data."""
        if self._scan_ranges is None:
            # No scan has arrived yet. The node only calls step() from its scan
            # callback so this is unreachable there, but a fixed-rate caller
            # can beat the first scan.
            return DriveCommand(0.0, self.crawl_speed, 0.0)

        # A fixed-rate caller can beat the next scan; replaying the last command
        # is better than solving the same scan twice.
        if self.t_scan is not None and self.t_scan == self._planned_scan_t:
            return self._last_planned
        self._planned_scan_t = self.t_scan

        ranges, angles = self.preprocess()
        steering, speed = self.plan(ranges, angles)
        self._last_planned = DriveCommand(float(steering), float(speed), 0.0)
        return self._last_planned

    def note_published(self, cmd: DriveCommand, t_sim=None):
        """Record what actually went on the wire."""
        self._last_cmd = cmd
        self._last_publish_t = t_sim

    def preprocess(self):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(self._scan_ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = self._scan_angle_min + np.arange(len(ranges)) * self._scan_angle_inc
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """MPC-based planning function.

        Uses Model Predictive Control to compute optimal steering and speed
        based on vehicle dynamics, reference path, and current state.
        """
        # 1. Restrict view to forward field of view [-100 deg, +100 deg] (~ -1.75 to +1.75 rad)
        fov_mask = (angles >= -1.75) & (angles <= 1.75)
        fov_ranges = ranges[fov_mask]
        fov_angles = angles[fov_mask]

        if len(fov_ranges) == 0:
            # No valid LiDAR data - fallback to crawl speed
            return 0.0, self.crawl_speed

        # 2. Basic obstacle avoidance - keep existing safety logic for robustness
        # Locate nearest obstacle point inside FOV
        min_idx = np.argmin(fov_ranges)
        min_dist = fov_ranges[min_idx]

        # Create a safety bubble around nearest obstacle (radius ~0.35m)
        car_radius = 0.35
        bubble_rad = math.atan2(car_radius, max(min_dist, 0.1))

        proc_ranges = fov_ranges.copy()
        bubble_mask = np.abs(fov_angles - fov_angles[min_idx]) <= bubble_rad
        proc_ranges[bubble_mask] = 0.0

        # 3. Find the largest continuous gap beyond threshold distance
        gap_threshold = 1.5  # meters
        free_space = proc_ranges > gap_threshold

        if not np.any(free_space):
            # Fallback: steer towards furthest visible point if no clear gap
            best_idx = np.argmax(proc_ranges)
            target_angle = fov_angles[best_idx]
        else:
            # Detect contiguous gap ranges
            bounded = np.pad(free_space.astype(int), (1, 1), 'constant')
            diffs = np.diff(bounded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            # Choose the widest gap
            gap_lengths = ends - starts
            max_gap_idx = np.argmax(gap_lengths)

            start_i = starts[max_gap_idx]
            end_i = ends[max_gap_idx]

            # Target the center of the largest gap
            gap_angles = fov_angles[start_i:end_i]
            target_angle = (gap_angles[0] + gap_angles[-1]) / 2.0

        # 4. Reactive steering: steer at the middle of the widest gap. Used as
        #    the MPC's warm-start guess and as the fallback if the MPC fails.
        reactive_steering = float(np.clip(0.85 * target_angle, -0.4, 0.4))

        # 5. Reactive speed: a crude clearance heuristic, kept as an upper
        #    bound and fallback only. The real speed target comes from the
        #    curvature profile once the MPC is in the loop.
        front_mask = (angles >= -0.2) & (angles <= 0.2)
        front_dist = np.mean(ranges[front_mask]) if np.any(front_mask) else 1.0

        if abs(reactive_steering) > 0.25 or front_dist < 2.0:
            reactive_speed = 2.0
        elif front_dist > 5.0 and abs(reactive_steering) < 0.1:
            reactive_speed = 4.5
        else:
            reactive_speed = 3.0

        # 6. Use MPC when we have both a path and a state. Without a reference
        #    path the MPC has nothing to track and would drive the car to the
        #    origin, so stay reactive in that case.
        if self.position is not None and self.reference_path is not None:
            try:
                # Current state: [x, y, yaw, velocity, beta_sideslip]
                # Beta (sideslip) is assumed 0 for simplified model
                current_state = np.array([
                    self.position[0],  # x
                    self.position[1],  # y
                    self.yaw,          # yaw
                    self.speed,        # velocity
                    0.0                # beta_sideslip (assumed zero)
                ])

                # Warm-start the optimizer from the reactive command.
                current_control = np.array([reactive_steering, 0.0])

                mpc_control, diagnostics = self.mpc_controller.get_control(
                    current_state, current_control)

                steering = float(mpc_control[0])
                accel = float(mpc_control[1])

                # /drive takes a SPEED, not an acceleration -- the simulator
                # closes that loop itself. The MPC plans in acceleration, so
                # project one step forward.
                speed = max(self.speed + accel * self.dt_measured, 0.0)

                # Never let the MPC talk the car into a wall: the reactive
                # clearance heuristic is a genuine safety bound, so take the
                # lower of the two.
                speed = min(speed, reactive_speed)
                steering = float(np.clip(steering, -self.steer_limit, self.steer_limit))

                if hasattr(self, '_diag_counter'):
                    self._diag_counter += 1
                else:
                    self._diag_counter = 0

                if self._diag_counter % 50 == 0:  # Every ~2 s at 40 Hz
                    self.log.info(
                        f'MPC: steer={steering:.3f}, speed={speed:.2f} '
                        f'(cur {self.speed:.2f}), solve={diagnostics["solve_time_ms"]:.1f}ms')

            except Exception as e:
                self.log.warn(f'MPC failed, using reactive: {e}')
                steering = reactive_steering
                speed = reactive_speed
        else:
            # No odometry or no reference path - use reactive
            steering = reactive_steering
            speed = reactive_speed

        return steering, speed
