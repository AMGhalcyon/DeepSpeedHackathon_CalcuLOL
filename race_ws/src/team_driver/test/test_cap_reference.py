"""Sub-10 Step 3: the emergency cap's corridor continues along the reference.

The MPC's plan covers 0.5 s. Continued straight past its end, the corridor runs
into the outside wall of every corner the plan has not reached yet, so the cap
braked at corner entries the profile already slows for (s~27.8, 48.9, 72.8 on
the icra26 line). With `cap_follows_reference` the corridor bends with the
reference past the plan's end instead, keeping the plan's offset from it.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import driver_core  # noqa: E402
import driver_params  # noqa: E402
import longitudinal  # noqa: E402
from test_step4_safety import (ANGLE_MIN, ANGLE_INC, ANGLES, CFG, _cast,  # noqa: E402
                               _core_on, _hairpin, _lidar_scan)

ON = driver_core.DriverConfig(cap_follows_reference=True)
SHORT_PLAN = _hairpin()[:3]          # 0.1 m of the straight: the bend is past its end
V = 6.0


def _outside_wall():
    """Straight ahead of the hairpin's entry: the outside wall of its bend."""
    return _cast((longitudinal.LIDAR_FORWARD, 0.0), 0.0, [((2.0, -3.0), (2.0, 0.4))])


def _wall_across_the_bend():
    """Across the hairpin at its far side (x 1.3, y 1.0): in the bend's own path."""
    return _lidar_scan(0.0, 0.0, 0.0, [((0.8, 1.0), (3.0, 1.0))])


def _continued(core, plan, length):
    """The corridor `_continued_plan` builds, without its hop."""
    return core._continued_plan(plan, length)[0]


def _step(cfg, ranges, arc_of=lambda _s: 0.0, plan=SHORT_PLAN):
    core, _ = _core_on(_hairpin(), np.full(len(_hairpin()), 10.0), arc_of, cfg=cfg,
                       plan_of=lambda _s: plan)
    core.set_odom(0.0, 0.0, 0.0, V, t_sim=0.0)
    core.set_scan(ranges, ANGLE_MIN, ANGLE_INC, t_sim=0.0)
    cmd = core.step(t_sim=0.0)
    return cmd, core.conditioning_stats().emergency_binds


class CorridorTest(unittest.TestCase):

    def test_off_a_short_plan_is_continued_straight_into_the_outside_wall(self):
        cmd, binds = _step(CFG, _outside_wall())
        self.assertLess(cmd.speed, V)
        self.assertEqual(binds, 1)

    def test_on_the_corridor_bends_with_the_reference_past_the_plan(self):
        cmd, binds = _step(ON, _outside_wall())
        self.assertAlmostEqual(cmd.speed, V, places=9)
        self.assertEqual(binds, 0)

    def test_on_a_wall_in_the_bends_own_path_still_caps_the_car(self):
        # The straight corridor never sees this wall; the bent one must.
        _, binds_off = _step(CFG, _wall_across_the_bend())
        cmd, binds_on = _step(ON, _wall_across_the_bend())
        self.assertEqual(binds_off, 0)
        self.assertEqual(binds_on, 1)
        self.assertLess(cmd.speed, V)

    def test_on_with_no_arc_position_the_corridor_stays_straight(self):
        # An MPC that reports no arc: nothing to continue along, so fail closed.
        core, mpc = _core_on(_hairpin(), np.full(len(_hairpin()), 10.0), lambda _s: 0.0,
                             cfg=ON, plan_of=lambda _s: SHORT_PLAN)
        mpc.get_control = lambda _state: (np.array([0.0, 0.0]), {'solve_time_ms': 0.0})
        mpc._plan = SHORT_PLAN
        core.set_odom(0.0, 0.0, 0.0, V, t_sim=0.0)
        core.set_scan(_outside_wall(), ANGLE_MIN, ANGLE_INC, t_sim=0.0)
        self.assertLess(core.step(t_sim=0.0).speed, V)
        self.assertEqual(core.conditioning_stats().emergency_binds, 1)


class ContinuedPlanTest(unittest.TestCase):

    def _circle_core(self, arc, outward=0.0, yaw_error=0.0, beta=0.0):
        """A core on a radius-5 circle, the car `outward` of it at `arc`, `yaw_error` off it."""
        phi = np.arange(0.0, 2.0 * math.pi, 0.01)
        circle = np.column_stack([5.0 * np.cos(phi), 5.0 * np.sin(phi)])
        core, _ = _core_on(circle, np.full(len(circle), 10.0), lambda _s: arc, cfg=ON)
        core._solved_arc = arc
        ref = longitudinal.ClosedPath(circle)
        p = ref.point_at(arc)
        tangent = ref.point_at(arc + 0.01) - p
        x, y = p * (1.0 + outward / np.linalg.norm(p))
        core.set_odom(x, y, math.atan2(tangent[1], tangent[0]) + yaw_error, V, beta=beta,
                      t_sim=0.0)
        return core, ref

    @staticmethod
    def _tail_arcs(plan, tail, s0):
        """Reference arcs the tail should sit at: past the plan's end, one spacing apart."""
        extent = float(np.sum(np.linalg.norm(np.diff(plan, axis=0), axis=1)))
        return s0 + extent + driver_core.CAP_CONTINUATION_SPACING * np.arange(1, len(tail) + 1)

    def test_the_continuation_keeps_the_cars_offset_from_the_reference(self):
        core, ref = self._circle_core(0.0, outward=0.3)  # 0.3 m outside the circle at s 0
        offset = np.array([0.3, 0.0])
        plan = ref.points_at(np.array([0.0, 0.1])) + offset
        tail = _continued(core, plan, 3.0)[len(plan):]
        self.assertGreater(len(tail), 0)
        arcs = self._tail_arcs(plan, tail, 0.0)
        np.testing.assert_allclose(tail, ref.points_at(arcs) + offset, atol=1e-3)
        self.assertGreaterEqual(arcs[-1], 3.0)

    def test_the_continuation_starts_where_the_plan_ends_not_where_it_starts(self):
        core, ref = self._circle_core(0.0)
        plan = ref.points_at(np.arange(0.0, 1.0 + 1e-9, 0.1))     # 1 m along the reference
        tail = _continued(core, plan, 3.0)[len(plan):]
        self.assertGreater(len(tail), 0)
        np.testing.assert_allclose(tail, ref.points_at(self._tail_arcs(plan, tail, 0.0)),
                                   atol=1e-3)

    def test_a_straight_plan_from_a_car_on_the_line_is_still_continued_round_the_bend(self):
        # Move blocking holds the plan's last move, so at a corner entry it runs on
        # straight while the car turns in: the plan's own heading is no guide.
        core, ref = self._circle_core(0.0)
        start = ref.point_at(0.0)
        plan = np.array([start, start + [0.0, 1.0], start + [0.0, 2.0]])   # the tangent, held
        tail = _continued(core, plan, 4.0)[len(plan):]
        np.testing.assert_allclose(tail, ref.points_at(self._tail_arcs(plan, tail, 0.0)),
                                   atol=1e-3)

    def test_the_continuation_wraps_past_the_end_of_the_lap(self):
        _, ref = self._circle_core(0.0)
        s0 = ref.length - 0.6
        core, _ = self._circle_core(s0)
        plan = ref.points_at(np.array([s0, s0 + 0.1]))
        tail = _continued(core, plan, 2.0)[len(plan):]
        arcs = self._tail_arcs(plan, tail, s0)
        self.assertGreater(arcs[-1], ref.length + 1.0)  # well into the next lap
        np.testing.assert_allclose(tail, ref.points_at(arcs % ref.length), atol=1e-3)

    def test_a_car_heading_off_the_reference_stays_straight(self):
        # Pointed at the infield: the car is not about to follow the reference,
        # so the corridor must not bend away from what is ahead of it.
        core, ref = self._circle_core(
            0.0, yaw_error=driver_core.CAP_CONTINUATION_MAX_HEADING + 0.05)
        plan = ref.points_at(np.array([0.0, 0.1]))
        np.testing.assert_array_equal(_continued(core, plan, 3.0), plan)

    def test_a_yaw_a_turn_round_is_the_same_heading(self):
        # Nothing wraps odometry's yaw: +2 pi is still along the line.
        core, ref = self._circle_core(0.0, yaw_error=2.0 * math.pi)
        plan = ref.points_at(np.array([0.0, 0.1]))
        self.assertGreater(len(_continued(core, plan, 3.0)), len(plan))

    def test_a_sliding_car_stays_straight_whatever_its_yaw(self):
        # Yaw on the line, but moving 0.35 rad off it: the sideslip counts.
        core, ref = self._circle_core(
            0.0, beta=-(driver_core.CAP_CONTINUATION_MAX_HEADING + 0.05))
        plan = ref.points_at(np.array([0.0, 0.1]))
        np.testing.assert_array_equal(_continued(core, plan, 3.0), plan)

    def test_a_car_far_off_the_line_stays_straight(self):
        core, ref = self._circle_core(
            0.0, outward=driver_core.CAP_CONTINUATION_MAX_OFFSET + 0.05)
        plan = ref.points_at(np.array([0.0, 0.1]))
        np.testing.assert_array_equal(_continued(core, plan, 3.0), plan)

    def test_a_plan_already_as_long_as_needed_is_left_alone(self):
        core, ref = self._circle_core(0.0)
        plan = np.array([ref.point_at(s) for s in np.arange(0.0, 4.0, 0.1)])
        np.testing.assert_array_equal(_continued(core, plan, 2.0), plan)

    def test_off_the_plan_is_left_alone(self):
        core, ref = self._circle_core(0.0)
        core.cfg = CFG
        plan = np.array([ref.point_at(0.0), ref.point_at(0.5)])
        np.testing.assert_array_equal(_continued(core, plan, 3.0), plan)


class BridgeTest(unittest.TestCase):
    """A straight plan ends off the line; the hop back to it is no travel."""

    R = 2.5
    PHI = np.arange(0.0, 2.0 * math.pi, 0.01)
    CIRCLE = np.column_stack([R * np.cos(PHI), R * np.sin(PHI)])
    REF = longitudinal.ClosedPath(CIRCLE)

    def _cap(self, plan):
        core, _ = _core_on(self.CIRCLE, np.full(len(self.CIRCLE), 10.0), lambda _s: 0.0,
                           cfg=ON)
        core._solved_arc = 0.0
        core.set_odom(self.R, 0.0, math.pi / 2.0, 8.0, t_sim=0.0)
        p = self.REF.point_at(3.5)                               # a wall across the line
        ranges = _lidar_scan(self.R, 0.0, math.pi / 2.0, [(0.8 * p, 1.2 * p)])
        return core._path_cap(ranges, ANGLES, plan)

    def test_a_wall_past_a_straight_plan_is_as_near_as_past_one_on_the_line(self):
        on_line = self._cap(self.REF.points_at(np.arange(0.0, 2.0 + 1e-9, 0.1)))
        straight = self._cap(np.array([[self.R, 0.0], [self.R, 1.0], [self.R, 2.0]]))
        self.assertLess(on_line, CFG.v_max)
        # Counted as travel, the 0.9 m hop from (2.5, 2) back to the line lifts it ~0.6 m/s.
        self.assertLessEqual(straight, on_line + 0.1)


class PointsAtTest(unittest.TestCase):

    def test_points_at_matches_point_at_and_wraps(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        path = longitudinal.ClosedPath(square)
        arcs = np.array([0.0, 0.5, 1.5, 3.9, 4.5, -0.25])
        np.testing.assert_allclose(path.points_at(arcs),
                                   [path.point_at(s) for s in arcs], atol=1e-12)

    def test_points_at_refuses_non_finite_arcs(self):
        path = longitudinal.ClosedPath(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
        with self.assertRaises(ValueError):
            path.points_at(np.array([0.0, math.nan]))


class DefaultsTest(unittest.TestCase):

    def test_the_cap_follows_the_reference_only_when_asked(self):
        self.assertIs(driver_core.DriverConfig().cap_follows_reference, False)
        self.assertIs(driver_params.DEFAULTS['cap_follows_reference'], False)


if __name__ == '__main__':
    unittest.main()
