#!/usr/bin/env python3
"""The judged vehicle plant, offline and ROS-free.

`lap_sim.py` drives `VehicleModel.dynamics_discrete`, which applies the
commanded steering angle *instantly*. The judged plant does not. It runs a
saturated P servo on the steering angle:

    sv = clip(kp * (delta_cmd - delta), -sv_max, +sv_max)

with `kp = 10*sv_max/(s_max - s_min) = 38.196` and `sv_max = 3.2 rad/s`. So
steering is a STATE with a ~26 ms time constant that rate-saturates once the
error exceeds `sv_max/kp = 0.0838 rad`. A full-lock reversal takes 262 ms --
longer than the MPC's whole 200 ms horizon. That is the mechanism behind the
chatter, and a harness that applies steering instantly cannot see it. Every
weight swept against the old harness was tuned blind.

This module reproduces the plant by IMPORTING the judged code rather than
copying it. `external/f1tenth_gym` needs `numba` (only for `njit`, a pure
decorator here) and its package `__init__` pulls in `gymnasium` via `F110Env`.
Neither is installed. `_load_gym_dynamics` stubs `njit` and loads the dynamics
subpackage under a synthetic package name, bypassing those `__init__` files. No
line of the vehicle model is transcribed, so the harness cannot silently drift
from the judge -- the exact failure that made the previous harness useless.

Scans and collisions are NOT here. The judge's LiDAR is an exact analytic
ray/segment scanner (JAX) and its collision model is a 64-iteration impulse
solver (JAX, walls via scikit-image). None of those three packages is
installed. That work is Step 1b.
"""

import math
import os
import sys
import types

import numpy as np


def _install_numba_stub():
    """Make `from numba import njit` work without numba.

    `njit` is a compilation hint. Every kernel here is plain numpy that runs
    identically interpreted, so the stub returns the function untouched. It is
    installed only when numba is genuinely absent, so a later `pip install
    numba` silently upgrades to the real thing with no code change.

    Returns:
        True if a stub was installed, False if real numba was already importable.
    """
    if 'numba' in sys.modules:
        return False
    try:
        import numba  # noqa: F401
        return False
    except ImportError:
        pass

    def njit(*args, **kwargs):
        # Both spellings appear upstream: bare `@njit` and `@njit(cache=True)`.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    stub = types.ModuleType('numba')
    stub.njit = njit
    stub.jit = njit
    sys.modules['numba'] = stub
    return True


def _gym_envs_dir():
    """Locate `external/f1tenth_gym/f1tenth_gym/envs`.

    Walks up from this file rather than assuming a working directory, so the
    module imports the same whether it is run from the repo root, from the
    package dir, or from an installed `race_ws/install/...` tree.
    """
    node = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        node = os.path.dirname(node)
        candidate = os.path.join(
            node, 'external', 'f1tenth_gym', 'f1tenth_gym', 'envs')
        if os.path.isdir(candidate):
            return candidate
    raise ImportError(
        'external/f1tenth_gym/f1tenth_gym/envs not found walking up from '
        f'{os.path.abspath(__file__)}')


def _load_gym_dynamics():
    """Import the judged dynamics code, bypassing the gymnasium-bound package init.

    `f1tenth_gym/__init__.py` does `import gymnasium` and
    `f1tenth_gym/envs/__init__.py` does `from .f110_env import F110Env`, so a
    plain `import f1tenth_gym.envs.dynamic_models` drags in gymnasium. Both are
    registered here as synthetic namespace packages carrying only `__path__`,
    so their `__init__.py` never executes while relative imports inside the
    subpackage (`.utils`, `..kinematic`, and the absolute
    `f1tenth_gym.envs.dynamic_models.utils` in multi_body) still resolve.

    Returns:
        (dynamic_models_module, integrators_module).
    """
    _install_numba_stub()
    envs_dir = _gym_envs_dir()
    gym_dir = os.path.dirname(envs_dir)
    root_dir = os.path.dirname(gym_dir)

    for name, path in (('f1tenth_gym', gym_dir), ('f1tenth_gym.envs', envs_dir)):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [path]
            pkg.__package__ = name
            sys.modules[name] = pkg

    # multi_body/multi_body.py imports `f1tenth_gym.envs.dynamic_models.utils`
    # absolutely, which resolves through the synthetic packages above, but the
    # finder still needs the tree on sys.path.
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    import importlib
    dm = importlib.import_module('f1tenth_gym.envs.dynamic_models')
    integrators = importlib.import_module('f1tenth_gym.envs.integrators')
    return dm, integrators


_DM, _INTEGRATORS = _load_gym_dynamics()

# The judge's code, not a copy of it.
vehicle_dynamics_st = _DM.vehicle_dynamics_st
pid_steer = _DM.pid_steer
pid_accl = _DM.pid_accl
rk4_integration = _INTEGRATORS.rk4_integration
VehicleParameters = _DM.VehicleParameters
F1TENTH_VEHICLE_PARAMETERS = _DM.F1TENTH_VEHICLE_PARAMETERS

# State layout of the 7-state single-track model, for readers and for tests.
X, Y, DELTA, V, PSI, PSI_DOT, BETA = range(7)


class GymPlant:
    """The judged plant: saturated steering servo + single-track dynamics.

    Mirrors `F110Simulator.step` for one agent. The defaults match what the
    race bridge actually builds -- `gym_bridge.py:390-393` constructs
    `ControlConfig(longitudinal_mode=SPEED, steering_mode=STEERING_ANGLE)` and
    passes no delay, no command noise and no `steer_kp`, so all of those are
    inert in the scored run. They are still exposed, because if a future config
    turns any of them on they become a second, independent source of chatter.

    Attributes:
        state: (7,) float32 `[x, y, delta, v, psi, psi_dot, beta]` at the CoG.
        sv_last: Steering velocity actually applied on the last step, rad/s.
            `|sv_last| == sv_max` is a rate-saturated step -- the chatter signal.
        accl_last: Longitudinal acceleration actually applied, m/s^2.
    """

    def __init__(self, params=None, timestep=0.01, integrator_dt=0.01,
                 steer_kp=None, steer_delay_steps=0, throttle_delay_steps=0,
                 steer_noise_std=0.0, accl_noise_std=0.0, seed=None):
        self.params = params if params is not None else F1TENTH_VEHICLE_PARAMETERS
        self.params_array = self.params.to_array()

        self.timestep = float(timestep)
        self.integrator_dt = float(integrator_dt)
        # `timestep % integrator_dt` rejects exact multiples IEEE754 cannot
        # represent (0.03 % 0.01 is 0.00999...), so compare the ratio instead.
        ratio = self.timestep / self.integrator_dt
        # The redundant int() mirrors simulator.py:98 verbatim, so the two
        # can be diffed line for line.
        self.substeps = max(1, int(round(ratio)))  # noqa: RUF046
        if not np.isclose(ratio, self.substeps, rtol=0.0, atol=1e-9):
            raise ValueError(
                f'timestep ({self.timestep}) must be an integer multiple of '
                f'integrator_timestep ({self.integrator_dt}), got a ratio of {ratio}')

        if steer_kp is None:
            # action.py:47-48 -- saturated beyond 1/10th of the steering range.
            steer_kp = 10.0 * self.params.sv_max / (self.params.s_max - self.params.s_min)
        self.steer_kp = float(steer_kp)

        self.steer_delay_steps = int(steer_delay_steps)
        self.throttle_delay_steps = int(throttle_delay_steps)
        self.steer_noise_std = float(steer_noise_std)
        self.accl_noise_std = float(accl_noise_std)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros(7, dtype=np.float32)
        self.steer_cmd = 0.0
        self.speed_cmd = 0.0
        self.sv_last = 0.0
        self.accl_last = 0.0
        self.time = 0.0
        self._steer_buffer = None
        self._throttle_buffer = None
        self._reset_buffers()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _reset_buffers(self):
        self._steer_buffer = ([0.0] * self.steer_delay_steps
                              if self.steer_delay_steps > 0 else None)
        self._throttle_buffer = ([0.0] * self.throttle_delay_steps
                                 if self.throttle_delay_steps > 0 else None)

    def reset(self, pose=(0.0, 0.0, 0.0)):
        """Place the car at a base_link pose, at rest.

        `pose` is base_link (rear axle), as every ROS topic and launch
        parameter carries it; the plant state is at the CoG, so the pose is
        converted the way `gym_bridge._base_link_to_cog` does.
        """
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        lr = float(self.params.lr)
        state = np.zeros(7)
        state[X] = x + lr * math.cos(yaw)
        state[Y] = y + lr * math.sin(yaw)
        state[PSI] = yaw
        self.state = state.astype(np.float32)

        self.steer_cmd = 0.0
        self.speed_cmd = 0.0
        self.sv_last = 0.0
        self.accl_last = 0.0
        self.time = 0.0
        self.rng = np.random.default_rng(self.seed)
        self._reset_buffers()
        return self.state

    def reset_state(self, state):
        """Place the car by full 7-state vector, for unit tests."""
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (7,):
            raise ValueError(f'state must have shape (7,), got {state.shape}')
        self.state = state.astype(np.float32)
        return self.state

    # ------------------------------------------------------------------
    # One control cycle
    # ------------------------------------------------------------------
    def latch(self, steering_angle, speed):
        """Hold a /drive command until the next one arrives.

        The driver publishes at ~40 Hz and the plant steps at 100 Hz, so the
        bridge holds the last command across several steps -- this is where
        part of the loop's dead time lives. Steering is clipped to the physical
        lock, matching `gym_bridge.py:595`; speed is NOT clipped there, it is
        left to `pid_accl` and `accl_constraints` inside the plant.
        """
        self.steer_cmd = float(np.clip(steering_angle,
                                       self.params.s_min, self.params.s_max))
        self.speed_cmd = float(speed)

    def _push_delay(self, buffer, value):
        """Ring-buffer one command; returns the one that comes due now."""
        buffer.append(value)
        return buffer.pop(0)

    def step(self):
        """Advance one outer timestep.

        Mirrors `simulator.py:323-338`. The steering and longitudinal efforts
        are evaluated ONCE, from the pre-step state, and held constant across
        every RK4 sub-stage. That is not an optimisation: re-evaluating the
        servo inside the integrator is a different, better-damped plant, and it
        would hide the rate saturation this module exists to expose.
        """
        steer_command = self.steer_cmd
        accel_command = self.speed_cmd

        if self.steer_noise_std > 0.0:
            steer_command += float(self.rng.normal(0.0, self.steer_noise_std))
        if self.accl_noise_std > 0.0:
            accel_command += float(self.rng.normal(0.0, self.accl_noise_std))
        if self._throttle_buffer is not None:
            accel_command = self._push_delay(self._throttle_buffer, accel_command)
        if self._steer_buffer is not None:
            steer_command = self._push_delay(self._steer_buffer, steer_command)

        state = self.state
        # action.py steering_angle_action / speed_action, inlined to avoid
        # importing action.py, which needs gymnasium for an unused Box.
        sv = pid_steer(steer_command, state[DELTA], self.params.sv_max, self.steer_kp)
        accl = pid_accl(accel_command, state[V], self.params.a_max,
                        self.params.v_max, self.params.v_min)
        control_vector = np.array([sv, accl], dtype=np.float32)
        self.sv_last = float(sv)
        self.accl_last = float(accl)

        for _ in range(self.substeps):
            state = rk4_integration(vehicle_dynamics_st, state, control_vector,
                                    self.integrator_dt, self.params_array)
        state[PSI] = (state[PSI] + np.pi) % (2 * np.pi) - np.pi
        self.state = state.astype(np.float32)
        self.time += self.timestep
        return self.state

    # ------------------------------------------------------------------
    # Outputs, in the frames the driver actually receives
    # ------------------------------------------------------------------
    @property
    def base_link_pose(self):
        """`(x, y, yaw)` at the rear axle, as /ego_racecar/odom publishes it."""
        lr = float(self.params.lr)
        psi = float(self.state[PSI])
        return (float(self.state[X]) - lr * math.cos(psi),
                float(self.state[Y]) - lr * math.sin(psi),
                psi)

    @property
    def odom_twist(self):
        """`(linear.x, linear.y, angular.z)` as the bridge fills them.

        Sideslip and true yaw rate arrive free on this topic: `linear.y` is
        `v*sin(beta)`, so beta is recoverable, and `angular.z` is the measured
        yaw rate. Step 6's steering estimator is built on these.
        """
        v = float(self.state[V])
        beta = float(self.state[BETA])
        return (v * math.cos(beta), v * math.sin(beta), float(self.state[PSI_DOT]))

    @property
    def is_rate_saturated(self):
        """Whether the last step hit the steering-velocity limit."""
        return abs(self.sv_last) >= float(self.params.sv_max) - 1e-9
