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

    `driver_params.py` declares one node parameter per field here, with these
    defaults, so a forgotten `--driver-params` flag is a provable no-op rather
    than a silently different car.
    """

    crawl_speed: float = 1.0        # [m/s]
    max_range: float = 8.0          # [m] clip the scan here
    v_max: float = 3.0              # [m/s] speed cap on the profile
    lat_accel: float = 0.8          # [m/s^2] cornering budget
    a_brake: float = 0.8            # [m/s^2] braking budget
    steer_limit: float = 0.4        # [rad]
    # [m] Clearance the braking-distance cap keeps in hand. Swept 0.0-0.5 over
    # 10 harness laps: 0 collisions at every value and at stale 5/9/14, and each
    # 0.1 m cost ~1 s of lap time, so the smallest value was taken. The cap
    # still assumes only a_brake (0.8 m/s^2) of braking, far below what the
    # plant can do, so 0.0 is not zero reserve.
    stop_margin: float = 0.0
    k_steer_derate: float = 2.0     # speed cap in a turn: v_max / (1 + k |steer|)
    coast_steer_rate: float = 1.0   # [rad/s] steering decay toward straight while coasting
    accel_limit: float = 4.0        # [m/s^2] fastest the speed setpoint may rise


class DriveCommand(NamedTuple):
    """One command destined for /drive.

    `steering_angle_velocity` is the rate the conditioner implies,
    (steer - previous steer) / dt. The judged config runs the simulator in
    STEERING_ANGLE mode and ignores it, so it documents the command rather than
    drives the plant.
    """

    steering_angle: float
    speed: float
    steering_angle_velocity: float = 0.0


class ConditioningStats(NamedTuple):
    """How hard the output conditioner has had to work, since construction."""

    steps: int          # commands conditioned (coasts are counted separately)
    slew_binds: int     # of those, commands the slew limiter shortened
    coast_count: int    # commands replaced by `_coast()`

    @property
    def slew_bind_rate(self) -> float:
        return self.slew_binds / self.steps if self.steps else 0.0


DT_CLAMP_LOW = 0.5     # x nominal: a burst of scans must not zero the interval
DT_CLAMP_HIGH = 3.0    # x nominal: a dropped scan must not hand over a 1 s dt

PLANT_STEP_DT = 0.01            # [s] gym_bridge.py sim_timestep; one servo evaluation each
STEER_CORRECT_MIN_SPEED = 0.8   # [m/s] below this the yaw-rate inversion is ill-conditioned
STEER_CORRECT_TAU = 0.5         # [s] time constant of the yaw-rate leash (~5 % per 25 ms)
LATCH_TIME_TOL = 1e-6           # [s] stamps are float sums of 0.01; compare with slack
# [s] Sim time jumping back further than this is a reset (/initialpose makes
# the bridge call env.reset(), which zeroes sim_time and re-centres the wheel).
# Smaller backward steps are an out-of-order stamp and are ignored.
SIM_RESET_BACKSTEP = 0.1
RTF_SMOOTHING = 0.05            # EMA weight per control step for the live RTF
FRONT_HALF_ANGLE = 0.2          # [rad] the sector the braking-distance cap looks down
# The near end of that sector, not its mean: one long beam down a gap must not
# lift the cap.
FRONT_PERCENTILE = 10


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


class SteeringEstimator:
    """Where the judged servo has put the wheel, for the MPC's sixth state.

    Nothing measures the wheel angle, so it is predicted: the plant's own
    servo -- `pid_steer` then `steering_constraint`, evaluated once per 10 ms
    physics step with a hard clip, since this simulates the plant rather than
    being differentiated by anyone -- is replayed over the commands the driver
    published, each from the plant time it latched. A command does not act at
    the scan it was planned from: it reaches the plant a solve later. Replaying
    by latch time is what keeps the estimate on the same instant as the pose it
    is paired with, however stale that pose is.

    Odometry's yaw rate leashes the prediction against drift. Inverting the
    kinematic yaw rate gives delta_meas = atan(L * psi_dot / (v cos beta)),
    well conditioned only at speed, and the estimate is pulled toward it with a
    ~0.5 s time constant. The predictor is primary; a stronger pull would feed
    yaw-rate noise and the tyres' ~2 % understeer into the MPC's initial state.
    """

    def __init__(self, params=None):
        p = params if params is not None else VehicleModel().params
        self.s_min = float(p.s_min)
        self.s_max = float(p.s_max)
        self.sv_max = float(p.sv_max)
        # action.py:47-48 -- the bridge passes no steer_kp, so the plant derives it.
        self.kp = 10.0 * self.sv_max / (self.s_max - self.s_min)
        self.wheelbase = float(p.lf + p.lr)

        self.delta_hat = 0.0      # [rad] the car starts on the grid, wheel centred
        self.t = None             # [s] plant time delta_hat refers to
        self._cmd = 0.0           # command latched at self.t
        self._pending = []        # (t_latch, cmd) not yet reached, time-ordered
        self._residuals = []      # |delta_hat - delta_meas| before each correction
        self.reseeds = 0          # sim resets seen; others key their own memory off this

    def on_publish(self, delta_cmd, t_latch=None):
        """A command went on the wire and latches at plant time `t_latch`.

        Clipped to the lock exactly as gym_bridge.py:595 does. A command with
        no time, or one whose time the estimate has already passed, acts from
        the estimate's own instant: the past cannot be replayed differently.
        """
        cmd = float(np.clip(delta_cmd, self.s_min, self.s_max))
        if self.t is None or t_latch is None or t_latch <= self.t + LATCH_TIME_TOL:
            # The newest command wins: anything published earlier is moot.
            self._pending = []
            self._cmd = cmd
            return
        t_latch = float(t_latch)
        # A newer command supersedes older ones due at or after it.
        self._pending = [entry for entry in self._pending if entry[0] < t_latch]
        self._pending.append((t_latch, cmd))

    def predict_to(self, t):
        """Advance the estimate to plant time `t`. Returns the interval advanced."""
        if t is None:
            return 0.0
        if self.t is not None and t < self.t - SIM_RESET_BACKSTEP:
            self._reseed()
        if self.t is None:
            self.t = float(t)
            return 0.0
        start = self.t
        while self.t < t - LATCH_TIME_TOL:
            self._latch_due()
            h = min(PLANT_STEP_DT, t - self.t)
            self.delta_hat = self._servo_step(self.delta_hat, self._cmd, h)
            self.t += h
        # Resynchronise on the caller's clock so float sums never drift.
        self.t = max(self.t, float(t))
        self._latch_due()
        return self.t - start

    def correct(self, yaw_rate, v, beta, dt):
        """Pull the estimate toward the wheel angle the yaw rate implies."""
        if v < STEER_CORRECT_MIN_SPEED or dt <= 0.0:
            return
        delta_meas = math.atan(self.wheelbase * yaw_rate / (v * math.cos(beta)))
        self._residuals.append(abs(self.delta_hat - delta_meas))
        gain = 1.0 - math.exp(-dt / STEER_CORRECT_TAU)
        self.delta_hat = float(np.clip(
            self.delta_hat + gain * (delta_meas - self.delta_hat),
            self.s_min, self.s_max))

    def residual_p95(self):
        """p95 of |delta_hat - delta_meas| since the last reset; 0 if none."""
        if not self._residuals:
            return 0.0
        return float(np.percentile(self._residuals, 95))

    def reset_stats(self):
        self._residuals = []

    def _reseed(self):
        """The sim was reset: the wheel is centred and old commands are void."""
        self.delta_hat = 0.0
        self.t = None
        self._cmd = 0.0
        self._pending = []
        self.reseeds += 1

    def _latch_due(self):
        while self._pending and self._pending[0][0] <= self.t + LATCH_TIME_TOL:
            self._cmd = self._pending.pop(0)[1]

    def _servo_step(self, delta, cmd, h):
        """utils.py pid_steer + steering_constraint, one step of length h."""
        sv = min(max(self.kp * (cmd - delta), -self.sv_max), self.sv_max)
        if (delta <= self.s_min and sv <= 0.0) or (delta >= self.s_max and sv >= 0.0):
            sv = 0.0
        return delta + sv * h


class LatchClock:
    """Plant time at which a command published now will latch, live.

    The node's sim clock is updated by a /clock callback on the same
    single-threaded executor as the control timer, so it stands still while
    the MPC solves -- but the bridge free-runs and keeps stepping. A command
    therefore latches about `solve_wall * rtf` of plant time after the clock
    reading the node holds. The real-time factor is learnt from how far sim
    and wall time move between control steps. Transport lag is not modelled,
    so this is still a slight lower bound; the live residual log is the check.
    """

    def __init__(self):
        self.rtf = None
        self._last = None

    def update(self, t_sim, t_wall):
        """Feed one (sim, wall) clock pair per control step."""
        if self._last is not None:
            d_sim = t_sim - self._last[0]
            d_wall = t_wall - self._last[1]
            if d_sim > 0.0 and d_wall > 0.0:
                ratio = d_sim / d_wall
                self.rtf = (ratio if self.rtf is None
                            else self.rtf + RTF_SMOOTHING * (ratio - self.rtf))
        self._last = (t_sim, t_wall)

    def latch_time(self, t_sim_now, solve_wall_s):
        """[s] Plant time a command published after `solve_wall_s` will latch."""
        rtf = self.rtf if self.rtf is not None else 1.0
        return t_sim_now + solve_wall_s * rtf


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
        self.stop_margin = self.cfg.stop_margin
        self.k_steer_derate = self.cfg.k_steer_derate
        self.coast_steer_rate = self.cfg.coast_steer_rate
        self.accel_limit = self.cfg.accel_limit

        self.mpc_controller = (mpc_controller if mpc_controller is not None
                               else MPCController(vehicle_model=VehicleModel()))
        self.mpc_dt = self.mpc_controller.dt
        self._dt = DtTracker(self.mpc_dt)
        self._planned_scan_t = None
        self._last_planned = DriveCommand(0.0, 0.0, 0.0)
        vehicle = getattr(self.mpc_controller, 'vehicle_model', None)
        self.steering = SteeringEstimator(vehicle.params if vehicle is not None else None)

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]
        # Sideslip and yaw rate arrive free on /ego_racecar/odom. Both leash the
        # steering estimate; the MPC itself still assumes beta = 0.
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

        # The conditioner's own memory of its last output: the base the rate
        # limits step from. None until the first command, which starts from the
        # wheel estimate and the measured speed instead.
        self._cond_steer = None
        self._cond_speed = None
        self._seen_reseeds = 0
        self._cond_steps = 0
        self._slew_binds = 0
        self._coast_count = 0
        self._coasting = False

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

    @property
    def delta_hat(self):
        """[rad] Estimated wheel angle at the last planned pose's stamp."""
        return self.steering.delta_hat

    def steering_residual_summary(self):
        """p95 of |delta_hat - delta_meas| (yaw-rate inversion); clears the window."""
        p95 = self.steering.residual_p95()
        self.steering.reset_stats()
        return p95

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
            # No scan has arrived yet; the fixed-rate node can beat the first
            # one. Driving blind is worse than standing still.
            return self._coast(self.dt_measured, 'no scan yet')

        # A fixed-rate caller can beat the next scan; replaying the last command
        # is better than solving the same scan twice.
        if self.t_scan is not None and self.t_scan == self._planned_scan_t:
            return self._last_planned
        self._planned_scan_t = self.t_scan

        # Bring the wheel estimate to the pose's instant -- odom's stamp, which
        # can trail the scan by a tick -- so the two describe the same moment
        # when they reach the MPC together, and the yaw rate it is corrected
        # with is from that moment too.
        t_pose = self.t_odom if self.t_odom is not None else self.t_scan
        advanced = self.steering.predict_to(t_pose)
        self.steering.correct(self.yaw_rate, self.speed, self.beta, advanced)
        if self.steering.reseeds != self._seen_reseeds:
            # The sim was reset: the car is back on the grid with the wheel
            # centred, so the old setpoints are not where any limit steps from.
            self._seen_reseeds = self.steering.reseeds
            self._cond_steer = None
            self._cond_speed = None
            self._coasting = False

        ranges, angles = self.preprocess()
        planned = self.plan(ranges, angles)
        dt = self.dt_measured
        if planned is None:
            self._last_planned = self._coast(dt, 'no usable plan')
        else:
            steering, speed, speed_cap = planned
            self._last_planned = self._condition(steering, speed, dt, speed_cap)
        return self._last_planned

    def conditioning_stats(self) -> ConditioningStats:
        """Slew-limiter and coast counts since construction."""
        return ConditioningStats(self._cond_steps, self._slew_binds, self._coast_count)

    # ------------------------------------------------------------------
    # Output conditioning: the single path to the wire
    # ------------------------------------------------------------------
    def _previous_output(self):
        """(steer, speed) the next command's rate limits step from."""
        steer = self._cond_steer if self._cond_steer is not None else self.steering.delta_hat
        speed = self._cond_speed if self._cond_speed is not None else self.speed
        # A NaN here would be stored as the new memory and never wash out.
        return (steer if math.isfinite(steer) else 0.0,
                speed if math.isfinite(speed) else 0.0)

    def _issue(self, steer, speed, prev_steer, dt) -> DriveCommand:
        self._cond_steer = steer
        self._cond_speed = speed
        rate = (steer - prev_steer) / dt if dt > 0.0 else 0.0
        return DriveCommand(steer, speed, rate)

    def _condition(self, raw_steer, raw_speed, dt, speed_cap=math.inf) -> DriveCommand:
        """Hold a raw command to what the servo and the speed loop can follow.

        The steering step is limited to exactly `sv_max * dt`: at that rate the
        limiter only removes motion the wheel could not have followed, so it
        adds no lag. Tighter would remove motion the wheel could have followed.
        The speed setpoint is rate-limited because /drive speed feeds a second
        loop (pid_accl), and a step in it is a step in demanded acceleration.

        `speed_cap` is the braking-distance bound and is applied last, as a
        hard ceiling. It is a safety bound, so the rate limit must not delay it:
        clearance can drop in a step (a wall comes into view) while the braking
        rate limit would take half a second to follow.
        """
        if not (math.isfinite(raw_steer) and math.isfinite(raw_speed)):
            return self._coast(dt, 'non-finite command')
        self._coasting = False

        prev_steer, prev_speed = self._previous_output()
        reach = self.steering.sv_max * dt
        target = float(np.clip(raw_steer, -self.steer_limit, self.steer_limit))
        steer = float(np.clip(target, prev_steer - reach, prev_steer + reach))
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))
        speed = float(np.clip(raw_speed,
                              prev_speed - self.a_brake * dt,
                              prev_speed + self.accel_limit * dt))

        self._cond_steps += 1
        if steer != target:
            self._slew_binds += 1
        return self._issue(steer, max(min(speed, speed_cap), 0.0), prev_steer, dt)

    def _coast(self, dt, why='') -> DriveCommand:
        """Degrade smoothly when there is no usable plan: never snap.

        Steering unwinds toward straight at `coast_steer_rate` and speed is shed
        at `a_brake`. The old fallbacks jumped the wheel to 0.0 from wherever it
        was, at the moment the controller had admitted it did not know what was
        happening.
        """
        prev_steer, prev_speed = self._previous_output()
        decay = self.coast_steer_rate * dt
        steer = prev_steer - math.copysign(min(abs(prev_steer), decay), prev_steer)
        speed = max(prev_speed - self.a_brake * dt, 0.0)

        self._coast_count += 1
        if not self._coasting:
            self._coasting = True
            self.log.warn(f'Coasting ({why or "no usable command"}); '
                          f'{self._coast_count} coast steps so far')
        return self._issue(steer, speed, prev_steer, dt)

    def _speed_cap(self, ranges, angles, steer) -> float:
        """[m/s] Upper bound on speed from stopping distance and steering.

        A braking-distance bound, so it has no steps: it falls as the clearance
        falls, and falls at about `a_brake` along the braking curve.
        """
        front = ranges[np.abs(angles) <= FRONT_HALF_ANGLE]
        clearance = float(np.percentile(front, FRONT_PERCENTILE)) if front.size else 1.0
        d_stop = max(clearance - self.stop_margin, 0.0)
        v_brake = math.sqrt(2.0 * self.a_brake * d_stop)
        v_turn = self.v_max / (1.0 + self.k_steer_derate * abs(steer))
        return float(np.clip(min(v_brake, v_turn), self.crawl_speed, self.v_max))

    def note_published(self, cmd: DriveCommand, t_sim=None):
        """Record what actually went on the wire.

        `t_sim` is the plant time the command takes effect -- when it is
        published, not the stamp of the scan it was planned from. The steering
        estimate replays the servo from exactly that instant.
        """
        self._last_cmd = cmd
        self._last_publish_t = t_sim
        self.steering.on_publish(cmd.steering_angle, t_sim)

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
        based on vehicle dynamics, reference path, and current state. Returns
        (steering, speed, speed_cap) before conditioning, or None when there is
        nothing usable to plan from and the caller should coast.
        """
        # 1. Restrict view to forward field of view [-100 deg, +100 deg] (~ -1.75 to +1.75 rad)
        fov_mask = (angles >= -1.75) & (angles <= 1.75)
        fov_ranges = ranges[fov_mask]
        fov_angles = angles[fov_mask]

        if len(fov_ranges) == 0:
            # No valid LiDAR data.
            return None

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
        #    the fallback if the MPC fails; the MPC keeps its own command memory.
        reactive_steering = float(np.clip(0.85 * target_angle, -0.4, 0.4))

        # 5. Use MPC when we have both a path and a state. Without a reference
        #    path the MPC has nothing to track and would drive the car to the
        #    origin, so stay reactive in that case. Either way the speed is
        #    bounded by `_speed_cap`, a braking-distance bound with no steps.
        if self.position is not None and self.reference_path is not None:
            try:
                # Current state: [x, y, yaw, velocity, beta_sideslip, wheel]
                # Beta (sideslip) is assumed 0 for simplified model
                current_state = np.array([
                    self.position[0],  # x
                    self.position[1],  # y
                    self.yaw,          # yaw
                    self.speed,        # velocity
                    0.0,               # beta_sideslip (assumed zero)
                    self.steering.delta_hat,  # wheel angle, estimated
                ])

                mpc_control, diagnostics = self.mpc_controller.get_control(
                    current_state)

                steering = float(mpc_control[0])
                accel = float(mpc_control[1])

                # /drive takes a SPEED, not an acceleration -- the simulator
                # closes that loop itself. The MPC plans in acceleration, so
                # project one step forward.
                mpc_speed = max(self.speed + accel * self.dt_measured, 0.0)
                steering = float(np.clip(steering, -self.steer_limit, self.steer_limit))

                if hasattr(self, '_diag_counter'):
                    self._diag_counter += 1
                else:
                    self._diag_counter = 0

                if self._diag_counter % 50 == 0:  # Every ~2 s at 40 Hz
                    self.log.info(
                        f'MPC: steer={steering:.3f}, speed={mpc_speed:.2f} '
                        f'(cur {self.speed:.2f}), solve={diagnostics["solve_time_ms"]:.1f}ms')

            except Exception as e:
                self.log.warn(f'MPC failed: {e}')
                return None

            # The braking-distance cap below stops the MPC talking the car into
            # a wall; _condition applies it as a ceiling.
            speed = mpc_speed
        else:
            # No odometry or no reference path - use reactive
            steering = reactive_steering
            speed = self.v_max

        return steering, speed, self._speed_cap(ranges, angles, steering)
