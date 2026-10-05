"""The MPC's reference window is paced by speeds the plant can reach.

Stock, state k sits at s0 + t_k * v_car, so while the car accelerates every
position reference is behind where the plan wants to be and the position cost
(q_pos 4x q_vel) holds the car back. With a brake limit set, the window marches
v_k = clip(profile(s_k), v_{k-1} - a_down*h, v_{k-1} + ceiling(v_{k-1})*h) and
s_k integrates v_k.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

from mpc_controller import MPCController  # noqa: E402
from vehicle_model import SpeedLoop, VehicleModel  # noqa: E402

HORIZON = 8
DT = 0.025
A_DOWN = 6.0
LOOP = SpeedLoop.of()
LAP = 400.0
SPACING = 0.2


def _controller(profile, brake_limit=A_DOWN, dt_schedule=None):
    """A closed circular lap of LAP metres; `profile(arc)` is the speed at each waypoint."""
    n = int(LAP / SPACING)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = LAP / (2.0 * np.pi)
    path = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    mpc = MPCController(vehicle_model=VehicleModel(), prediction_horizon=HORIZON,
                        control_horizon=HORIZON, dt=DT, dt_schedule=dt_schedule)
    mpc.set_reference_path(path, np.array([profile(s) for s in arc]),
                           brake_limit=brake_limit)
    return mpc


def _window_steps(mpc, s_start, car_speed):
    """(per-step mean speed, step lengths h) of the window."""
    mpc._reference_window(s_start, HORIZON + 1, car_speed=car_speed)
    h = np.diff(mpc._time_offsets(HORIZON + 1))
    return np.diff(mpc._window_s) / h, h


class ReachablePaceTest(unittest.TestCase):

    def test_accelerates_at_the_plant_ceiling_on_a_straight(self):
        mpc = _controller(lambda s: 12.0)
        pace, h = _window_steps(mpc, 10.0, car_speed=3.0)
        # first step: trapezoid from 3 m/s at the full accel limit
        self.assertAlmostEqual(pace[0], 3.0 + 0.5 * LOOP.a_max * h[0], places=9)
        self.assertTrue(np.all(np.diff(pace) > 0.0))              # still speeding up
        self.assertTrue(np.all(pace * h > 3.0 * h))               # ahead of constant pace

    def test_never_runs_past_the_profile(self):
        mpc = _controller(lambda s: 4.0)
        pace, _ = _window_steps(mpc, 10.0, car_speed=3.9)
        self.assertTrue(np.all(pace <= 4.0 + 1e-9))

    def test_shrinks_no_faster_than_a_down_before_a_slower_section(self):
        # 12 m/s, then a 2 m/s corner 0.3 m ahead (the whole window is only 0.2 s)
        mpc = _controller(lambda s: 2.0 if 10.3 <= s <= 60.0 else 12.0)
        pace, h = _window_steps(mpc, 10.0, car_speed=10.0)
        # mean speed over a step moves by half of each neighbouring step's change
        slew = np.diff(pace) / (0.5 * (h[1:] + h[:-1]))
        self.assertTrue(np.all(slew >= -A_DOWN - 1e-9))
        self.assertAlmostEqual(slew.min(), -A_DOWN, places=9)      # and the limit binds
        self.assertLess(pace[-1], pace[0])                         # it does slow down

    def test_power_ceiling_binds_above_the_switch_speed(self):
        v0 = 14.0
        self.assertGreater(v0, LOOP.v_switch)
        mpc = _controller(lambda s: 20.0)
        pace, h = _window_steps(mpc, 10.0, car_speed=v0)
        v1 = v0 + LOOP.accel_ceiling(v0) * h[0]
        self.assertAlmostEqual(pace[0], 0.5 * (v0 + v1), places=9)
        self.assertLess(LOOP.accel_ceiling(v0), LOOP.a_max)

    def test_window_wraps_at_the_finish_line(self):
        mpc = _controller(lambda s: 8.0)
        window = mpc._reference_window(mpc._path_len - 0.2, HORIZON + 1, car_speed=8.0)
        self.assertGreater(np.hypot(*(window[-1, :2] - window[0, :2])), 0.2)
        self.assertTrue(np.all(np.diff(mpc._window_s) > 0.0))
        self.assertGreater(mpc._window_s[-1], mpc._path_len)       # carried over the seam

    def test_the_profile_is_read_across_the_closing_segment(self):
        # Fast to the seam, then a 4 m/s start of lap. Reading the profile
        # without wrapping would clamp to the last waypoint's 12 m/s there.
        mpc = _controller(lambda s: 4.0 if s < 1.0 else 12.0)
        mpc._reference_window(mpc._path_len - 0.3, HORIZON + 1, car_speed=10.0)
        pace = np.diff(mpc._window_s) / np.diff(mpc._time_offsets(HORIZON + 1))
        self.assertLess(pace[-1], pace[0])

    def test_a_dt_schedule_marches_with_each_steps_own_length(self):
        schedule = [0.025, 0.025, 0.05, 0.05, 0.075, 0.075, 0.1, 0.1]
        mpc = _controller(lambda s: 12.0, dt_schedule=schedule)
        mpc._reference_window(10.0, HORIZON + 1, car_speed=3.0)
        s, v = [10.0], 3.0
        for h in schedule:
            v_next = min(12.0, v + LOOP.accel_ceiling(v) * h)
            s.append(s[-1] + 0.5 * (v + v_next) * h)
            v = v_next
        np.testing.assert_allclose(mpc._window_s, s, rtol=1e-12)

    def test_a_stopped_car_still_gets_an_advancing_window(self):
        mpc = _controller(lambda s: 8.0)
        pace, _ = _window_steps(mpc, 10.0, car_speed=0.0)
        self.assertTrue(np.all(pace > 0.0))


class StockPaceTest(unittest.TestCase):
    """Without the march the window is the old constant-pace one."""

    def test_no_brake_limit_keeps_constant_pace(self):
        mpc = _controller(lambda s: 12.0, brake_limit=None)
        pace, _ = _window_steps(mpc, 10.0, car_speed=5.0)
        np.testing.assert_allclose(pace, 5.0, rtol=1e-12)

    def test_no_car_speed_keeps_the_profile_mean_pace(self):
        mpc = _controller(lambda s: 12.0)
        pace, _ = _window_steps(mpc, 10.0, car_speed=None)
        np.testing.assert_allclose(pace, 12.0, rtol=1e-12)

    def test_no_speeds_keeps_constant_pace(self):
        mpc = _controller(lambda s: 12.0)
        mpc.reference_speeds = None
        pace, _ = _window_steps(mpc, 10.0, car_speed=5.0)
        np.testing.assert_allclose(pace, 5.0, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
