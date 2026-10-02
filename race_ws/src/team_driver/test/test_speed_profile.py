#!/usr/bin/env python3
"""The closed-loop, slip-budgeted speed profile (Phase 2, Step 1).

The profile must be drivable: braking is planned round the finish line, the
forward pass respects what the plant can accelerate, and the cornering limit is
the smaller of a slip budget (`a_lat_max`) and the steering lock (`v_lock`).
`v_lock` is pinned against the judged plant's own `vehicle_dynamics_st`.
"""

import math
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, 'team_driver'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import speed_profile as sp  # noqa: E402
from vehicle_model import F1TENTH_VEHICLE_PARAMETERS as VP  # noqa: E402

A_ACCEL = 9.51
A_BRAKE = 9.51
SPEC = dict(a_lat_max=5.0, lock_margin=0.9, a_accel_max=A_ACCEL,
            a_brake_max=A_BRAKE, v_max=20.0)


def _circle(radius, n=400):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _stadium(straight, radius, spacing=0.1):
    """A closed track: two straights joined by two semicircles."""
    n_s = int(straight / spacing)
    n_c = int(math.pi * radius / spacing)
    bottom = np.column_stack([np.linspace(0, straight, n_s, endpoint=False),
                              np.full(n_s, -radius)])
    a = np.linspace(-np.pi / 2, np.pi / 2, n_c, endpoint=False)
    right = np.column_stack([straight + radius * np.cos(a), radius * np.sin(a)])
    top = np.column_stack([np.linspace(straight, 0, n_s, endpoint=False),
                           np.full(n_s, radius)])
    b = np.linspace(np.pi / 2, 3 * np.pi / 2, n_c, endpoint=False)
    left = np.column_stack([radius * np.cos(b), radius * np.sin(b)])
    return np.vstack([bottom, right, top, left])


def _ds(path):
    return np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)


class TestCornering(unittest.TestCase):

    def test_circle_is_held_at_the_slip_budget_speed(self):
        radius = 3.0
        v = sp.speed_profile(_circle(radius), **SPEC)
        expected = math.sqrt(SPEC['a_lat_max'] * radius)
        np.testing.assert_allclose(v, expected, rtol=0.01)

    def test_tight_circle_is_limited_by_the_lock_not_the_slip_budget(self):
        radius = 0.9
        spec = {**SPEC, 'a_lat_max': 15.0}
        v = sp.speed_profile(_circle(radius, n=200), **spec)
        slip_limit = math.sqrt(spec['a_lat_max'] * radius)
        lock_limit = float(sp.lock_speed(np.array([1.0 / radius]), spec['lock_margin'])[0])
        self.assertLess(lock_limit, slip_limit)
        np.testing.assert_allclose(v, lock_limit, rtol=0.01)

    def test_line_tighter_than_the_lock_radius_is_floored_not_nan(self):
        v = sp.lock_speed(np.array([1.0 / 0.5]), 0.9)
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertGreaterEqual(float(v[0]), sp.MIN_CORNER_SPEED)

    def test_straight_runs_at_v_max(self):
        n = 200
        path = np.column_stack([np.linspace(0, 40, n), np.zeros(n)])
        v = sp.speed_profile(path, **{**SPEC, 'v_max': 6.0}, kappa=np.zeros(n))
        self.assertAlmostEqual(float(v.max()), 6.0, places=6)


def _plant_params_array():
    from plant_sim import F1TENTH_VEHICLE_PARAMETERS
    return F1TENTH_VEHICLE_PARAMETERS.to_array()


class TestLockSpeedMatchesThePlant(unittest.TestCase):
    """`v_lock` must reproduce the judged plant's steady turn to within 2 %."""

    def _steady_curvature(self, delta, speed):
        from plant_sim import (V, DELTA, PSI_DOT, vehicle_dynamics_st,
                               rk4_integration)
        params = _plant_params_array()
        state = np.zeros(7)
        state[DELTA], state[V] = delta, speed
        control = np.zeros(2)
        for _ in range(600):                      # 6 s at the plant's 10 ms step
            state = rk4_integration(vehicle_dynamics_st, state, control, 0.01, params)
        return state[PSI_DOT] / state[V]

    def test_v_lock_at_full_lock_turns_the_requested_radius(self):
        for radius in (0.9, 1.0, 1.5, 2.5):
            with self.subTest(radius=radius):
                kappa = 1.0 / radius
                v_lock = float(sp.lock_speed(np.array([kappa]), lock_margin=1.0)[0])
                self.assertGreater(v_lock, sp.MIN_CORNER_SPEED)
                measured = self._steady_curvature(VP.s_max, v_lock)
                self.assertAlmostEqual(measured / kappa, 1.0, delta=0.02)


class TestLongitudinal(unittest.TestCase):

    def setUp(self):
        self.path = _stadium(straight=15.0, radius=1.0)
        self.kappa = sp.closed_curvature(self.path)
        self.v = sp.speed_profile(self.path, **SPEC, kappa=self.kappa)
        self.ds = _ds(self.path)

    def test_brakes_before_the_hairpin_at_the_braking_limit(self):
        v, ds = self.v, self.ds
        decel = (v[:-1] ** 2 - v[1:] ** 2) / (2.0 * ds[:-1])
        self.assertLessEqual(float(decel.max()), A_BRAKE * 1.001)
        self.assertGreater(float(decel.max()), 0.9 * A_BRAKE,
                           'the profile never brakes at the limit it was given')
        corner = int(np.argmax(self.kappa))
        self.assertLess(float(v[corner]), float(v.max()))

    def test_forward_pass_respects_acceleration_and_the_power_fall_off(self):
        v, ds = self.v, self.ds
        accel = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds[:-1])
        allowed = np.minimum(A_ACCEL, VP.a_max * VP.v_switch / np.maximum(v[:-1], 0.1))
        self.assertTrue(np.all(accel <= allowed * 1.001),
                        f'worst excess {float((accel - allowed).max()):.4f} m/s^2')

    def test_fall_off_slows_acceleration_above_the_switch_speed(self):
        n = 3000
        path = np.column_stack([np.linspace(0, 400, n), np.zeros(n)])
        v = sp.speed_profile(path, **SPEC, kappa=np.zeros(n), closed=False)
        ds = _ds(path)[:-1]
        accel = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds)
        fast = v[:-1] > 1.2 * VP.v_switch
        slow = (v[:-1] < 0.8 * VP.v_switch) & (v[:-1] > 0.5)
        self.assertTrue(fast.any() and slow.any())
        self.assertGreater(float(accel[slow].mean()), 1.5 * float(accel[fast].mean()))


class TestWrap(unittest.TestCase):

    def test_corner_at_index_zero_is_braked_for_before_the_end_of_the_array(self):
        path = _stadium(straight=15.0, radius=1.0)
        kappa = sp.closed_curvature(path)
        shift = -int(np.argmax(kappa))
        path, kappa = np.roll(path, shift, axis=0), np.roll(kappa, shift)
        v = sp.speed_profile(path, **SPEC, kappa=kappa)
        approach = v[-int(8.0 / 0.1):]
        self.assertTrue(np.all(np.diff(approach) <= 1e-6),
                        'speed rises toward the corner at index 0: the braking '
                        'pass did not wrap')
        self.assertLess(float(v[-1]), float(v[-int(8.0 / 0.1)]))

    def test_open_path_does_not_wrap(self):
        n = 400
        x = np.linspace(0, 40, n)
        kappa = np.zeros(n)
        kappa[0] = 3.0
        v = sp.speed_profile(np.column_stack([x, np.zeros(n)]), **SPEC,
                             kappa=kappa, closed=False)
        self.assertGreater(float(v[-1]), 10.0)


class TestCurvatureIsClosed(unittest.TestCase):

    def test_circle_curvature_is_uniform_across_the_seam(self):
        kappa = sp.closed_curvature(_circle(2.0))
        np.testing.assert_allclose(kappa, 0.5, rtol=0.01)


class TestFloorMetrics(unittest.TestCase):

    def test_ideal_lap_time_of_a_circle(self):
        radius = 3.0
        path = _circle(radius, n=600)
        v = sp.speed_profile(path, **SPEC)
        expected = 2.0 * math.pi * radius / math.sqrt(SPEC['a_lat_max'] * radius)
        self.assertAlmostEqual(sp.ideal_lap_time(path, v), expected, delta=0.02 * expected)

    def test_rear_slip_follows_the_lateral_acceleration(self):
        # alpha_r = a_y / (mu * C_Sr * g) = a_y / 56.1
        v, kappa = 3.0, 1.0
        slip = sp.rear_slip(np.array([v]), np.array([kappa]))[0]
        self.assertAlmostEqual(slip, (v * v * kappa) / 56.1, delta=0.01 * slip)


class TestLapFloorTable(unittest.TestCase):

    def test_table_has_a_row_per_budget_and_lap_time_falls_as_the_budget_grows(self):
        import lap_floor
        rows = lap_floor.floor_table(_stadium(straight=10.0, radius=2.0),
                                     (5.0, 10.0, 15.0))
        self.assertEqual([r.a_lat_max for r in rows], [5.0, 10.0, 15.0])
        times = [r.lap_time for r in rows]
        self.assertTrue(all(a > b for a, b in zip(times, times[1:])), times)

    def test_shipped_centerline_is_found_and_closes_the_lap(self):
        import lap_floor
        path = lap_floor.load_path(lap_floor.default_centerline_path())
        length = float(np.sum(sp.segment_lengths(path, True)))
        self.assertAlmostEqual(length, 78.49, delta=0.5)


if __name__ == '__main__':
    unittest.main()
