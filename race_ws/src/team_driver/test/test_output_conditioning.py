"""Step 7: one conditioned path to the wire."""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import driver_core  # noqa: E402
import lap_sim  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

DT = 0.025
CFG = driver_core.DriverConfig()
SV_MAX = VehicleModel().params.sv_max
SLEW = SV_MAX * DT                      # 0.080 rad per 25 ms control step
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 181)
TOL = 1e-9


class _ScriptedMPC:
    """Returns whatever control the test sets, or raises."""

    dt = DT

    def __init__(self, steer=0.0, accel=0.0, error=None):
        self.steer = steer
        self.accel = accel
        self.error = error

    def speed_profile(self, **_kw):
        return np.full(4, 3.0)

    def set_reference_path(self, *_a):
        pass

    def get_control(self, _state):
        if self.error is not None:
            raise self.error
        return np.array([self.steer, self.accel]), {'solve_time_ms': 0.0}


def _core(**mpc_kw):
    mpc = _ScriptedMPC(**mpc_kw)
    core = driver_core.DriverCore(cfg=driver_core.DriverConfig(), mpc_controller=mpc)
    core.reference_path = np.zeros((4, 2))
    return core, mpc


def _feed(core, t, v=2.0, ranges=None, angle_min=-np.pi / 2):
    core.set_odom(0.0, 0.0, 0.0, v, beta=0.0, yaw_rate=0.0, t_sim=t)
    scan = np.full(181, 6.0) if ranges is None else ranges
    core.set_scan(scan, angle_min, np.pi / 180, t_sim=t)


def _drive(core, n, first=0, **feed):
    cmd = None
    for i in range(first, first + n):
        _feed(core, i * DT, **feed)
        cmd = core.step(t_sim=i * DT)
    return cmd


def _steady_core():
    """A core that has settled at steer 0.2, speed 2.0 on a clear track."""
    core, mpc = _core(steer=0.2)
    cmd = _drive(core, 5)
    assert abs(cmd.steering_angle - 0.2) < TOL and abs(cmd.speed - 2.0) < TOL
    return core, mpc


# ----------------------------------------------------------------------
# Slew limiter: exactly what the wheel can follow, never tighter
# ----------------------------------------------------------------------
class SlewLimiterTest(unittest.TestCase):

    def test_a_large_step_is_limited_to_the_rate_the_servo_can_follow(self):
        core, _ = _core()
        cmd = core._condition(0.3, 1.0, DT)
        self.assertAlmostEqual(cmd.steering_angle, SLEW, places=12)
        self.assertAlmostEqual(SLEW, 0.080, places=12)

    def test_a_step_the_servo_can_follow_passes_untouched(self):
        core, _ = _core()
        core._condition(0.0, 1.0, DT)
        cmd = core._condition(SLEW * 0.999, 1.0, DT)
        self.assertEqual(cmd.steering_angle, SLEW * 0.999)

    def test_the_limit_is_exact_so_a_large_demand_arrives_in_the_fewest_steps(self):
        core, _ = _core()
        path = [core._condition(0.3, 1.0, DT).steering_angle for _ in range(5)]
        self.assertEqual(path[3], 0.3)          # 0.08, 0.16, 0.24, 0.30
        for k in range(3):
            self.assertAlmostEqual(path[k], SLEW * (k + 1), places=12)

    def test_the_limit_scales_with_the_interval(self):
        core, _ = _core()
        cmd = core._condition(0.3, 1.0, 0.0125)
        self.assertAlmostEqual(cmd.steering_angle, SV_MAX * 0.0125, places=12)

    def test_a_negative_demand_is_limited_the_same_way(self):
        core, _ = _core()
        cmd = core._condition(-0.3, 1.0, DT)
        self.assertAlmostEqual(cmd.steering_angle, -SLEW, places=12)

    def test_the_command_never_leaves_the_steering_limit(self):
        core, _ = _core()
        for _ in range(20):
            cmd = core._condition(0.9, 1.0, DT)
        self.assertEqual(cmd.steering_angle, CFG.steer_limit)
        for _ in range(40):
            cmd = core._condition(-0.9, 1.0, DT)
        self.assertEqual(cmd.steering_angle, -CFG.steer_limit)

    def test_the_first_command_is_limited_from_the_wheel_not_from_zero(self):
        core, _ = _core()
        core.steering.delta_hat = 0.3
        cmd = core._condition(0.3, 1.0, DT)
        self.assertEqual(cmd.steering_angle, 0.3)

    def test_the_wire_message_carries_the_steering_velocity(self):
        core, _ = _core()
        first = core._condition(0.3, 1.0, DT)
        self.assertAlmostEqual(first.steering_angle_velocity, SV_MAX, places=9)
        second = core._condition(first.steering_angle - 0.01, 1.0, DT)
        self.assertAlmostEqual(second.steering_angle_velocity, -0.01 / DT, places=9)

    def test_published_steering_never_jumps_more_than_the_servo_can_move(self):
        core, _ = _core()
        rng = np.random.default_rng(0)
        prev = 0.0
        for raw in rng.uniform(-0.6, 0.6, size=300):
            steer = core._condition(float(raw), 1.0, DT).steering_angle
            self.assertLessEqual(abs(steer - prev), SLEW + TOL)
            prev = steer


# ----------------------------------------------------------------------
# Speed rate limit: no step in the setpoint
# ----------------------------------------------------------------------
class SpeedRateLimitTest(unittest.TestCase):

    def test_acceleration_demand_is_limited(self):
        core, _ = _core()
        core.speed = 1.0
        cmd = core._condition(0.0, 3.0, DT)
        self.assertAlmostEqual(cmd.speed, 1.0 + CFG.accel_limit * DT, places=12)

    def test_braking_demand_is_limited_to_the_braking_budget(self):
        core, _ = _core()
        core.speed = 2.0
        cmd = core._condition(0.0, 0.0, DT)
        self.assertAlmostEqual(cmd.speed, 2.0 - CFG.a_brake * DT, places=12)

    def test_speed_is_never_negative(self):
        core, _ = _core()
        core.speed = 0.01
        cmd = core._condition(0.0, -5.0, DT)
        self.assertEqual(cmd.speed, 0.0)

    def test_published_speed_has_no_discontinuities(self):
        core, _ = _core()
        core.speed = 1.0
        rng = np.random.default_rng(1)
        bound = max(CFG.accel_limit, CFG.a_brake) * DT + TOL
        prev = 1.0
        for raw in rng.uniform(0.0, 5.0, size=300):
            speed = core._condition(0.0, float(raw), DT).speed
            self.assertLessEqual(abs(speed - prev), bound)
            prev = speed


# ----------------------------------------------------------------------
# _coast: decay, never snap
# ----------------------------------------------------------------------
class CoastTest(unittest.TestCase):

    def _settled(self, steer):
        core, _ = _core()
        core.speed = 2.0
        for _ in range(10):
            core._condition(steer, 2.0, DT)
        return core

    def test_steering_decays_toward_straight_at_the_coast_rate(self):
        core = self._settled(0.2)
        cmd = core._coast(DT)
        self.assertAlmostEqual(cmd.steering_angle, 0.2 - CFG.coast_steer_rate * DT, places=12)

    def test_negative_steering_decays_toward_straight_too(self):
        core = self._settled(-0.2)
        cmd = core._coast(DT)
        self.assertAlmostEqual(cmd.steering_angle, -0.2 + CFG.coast_steer_rate * DT, places=12)

    def test_decay_stops_at_straight_instead_of_overshooting(self):
        core = self._settled(0.01)
        self.assertEqual(core._coast(DT).steering_angle, 0.0)
        self.assertEqual(core._coast(DT).steering_angle, 0.0)

    def test_speed_is_shed_at_the_braking_budget_not_snapped(self):
        core = self._settled(0.0)
        cmd = core._coast(DT)
        self.assertAlmostEqual(cmd.speed, 2.0 - CFG.a_brake * DT, places=12)

    def test_speed_stops_at_zero(self):
        core, _ = _core()
        core.speed = 0.01
        core._condition(0.0, 0.01, DT)
        self.assertEqual(core._coast(DT).speed, 0.0)

    def test_coasting_carries_the_steering_velocity(self):
        core = self._settled(0.2)
        cmd = core._coast(DT)
        self.assertAlmostEqual(cmd.steering_angle_velocity, -CFG.coast_steer_rate, places=9)

    def test_a_non_finite_command_is_coasted_not_published(self):
        core = self._settled(0.2)
        cmd = core._condition(float('nan'), 2.0, DT)
        self.assertAlmostEqual(cmd.steering_angle, 0.2 - CFG.coast_steer_rate * DT, places=12)
        self.assertEqual(core.conditioning_stats().coast_count, 1)
        cmd = core._condition(0.2, float('inf'), DT)
        self.assertTrue(math.isfinite(cmd.speed))
        self.assertEqual(core.conditioning_stats().coast_count, 2)


# ----------------------------------------------------------------------
# The fallbacks in step(): coast, never snap
# ----------------------------------------------------------------------
class FallbackPathTest(unittest.TestCase):

    def test_an_empty_field_of_view_coasts_instead_of_snapping(self):
        core, _ = _steady_core()
        _feed(core, 5 * DT, angle_min=2.0)          # every beam outside the FOV
        cmd = core.step(t_sim=5 * DT)
        self.assertAlmostEqual(cmd.steering_angle, 0.2 - CFG.coast_steer_rate * DT, places=9)
        self.assertAlmostEqual(cmd.speed, 2.0 - CFG.a_brake * DT, places=9)
        self.assertEqual(core.conditioning_stats().coast_count, 1)

    def test_an_mpc_exception_coasts_instead_of_falling_back_to_the_reactive_command(self):
        core, mpc = _steady_core()
        mpc.error = RuntimeError('solver blew up')
        cmd = _drive(core, 1, first=5)
        self.assertAlmostEqual(cmd.steering_angle, 0.2 - CFG.coast_steer_rate * DT, places=9)
        self.assertAlmostEqual(cmd.speed, 2.0 - CFG.a_brake * DT, places=9)
        self.assertEqual(core.conditioning_stats().coast_count, 1)

    def test_a_clean_run_never_coasts(self):
        core, _ = _steady_core()
        _drive(core, 20, first=5)
        self.assertEqual(core.conditioning_stats().coast_count, 0)

    def test_before_the_first_scan_the_car_does_not_move_blind(self):
        core, _ = _core()
        cmd = core.step()
        self.assertEqual((cmd.steering_angle, cmd.speed), (0.0, 0.0))
        self.assertEqual(core.conditioning_stats().coast_count, 1)

    def test_the_reactive_controller_is_slew_limited_too(self):
        core, _ = _core()
        core.reference_path = None                  # nothing for the MPC to track
        ranges = np.where((ANGLES > 0.5) & (ANGLES < 1.0), 8.0, 0.5)
        cmd = _drive(core, 1, ranges=ranges)
        self.assertAlmostEqual(cmd.steering_angle, SLEW, places=9)

    def test_the_planned_command_carries_the_steering_velocity(self):
        core, _ = _core(steer=0.3)
        cmd = _drive(core, 1)
        self.assertAlmostEqual(cmd.steering_angle_velocity, SV_MAX, places=9)

    def test_a_sim_reset_forgets_the_old_command(self):
        # After a collision the bridge respawns the car: wheel centred, v = 0,
        # sim time back at zero. The rate limits must step from that, not from
        # the setpoint the car had before.
        core, _ = _core(steer=0.2)
        _drive(core, 5, first=400)                  # settled at steer 0.2, speed 2.0
        _feed(core, 0.0, v=0.0)                     # sim time jumps back to zero
        cmd = core.step(t_sim=0.0)
        self.assertEqual(cmd.speed, 0.0)
        # One step from the centred wheel (the backward stamp makes this step's
        # interval the clamped minimum, so less than a full 25 ms of reach).
        self.assertGreater(cmd.steering_angle, 0.0)
        self.assertLessEqual(cmd.steering_angle, SLEW + TOL)

    def test_an_ordinary_run_is_not_mistaken_for_a_reset(self):
        core, _ = _steady_core()
        cmd = _drive(core, 1, first=5)
        self.assertAlmostEqual(cmd.steering_angle, 0.2, places=9)
        self.assertAlmostEqual(cmd.speed, 2.0, places=9)

    def test_a_poisoned_starting_state_does_not_poison_the_wire(self):
        core, _ = _core()
        core.speed = float('nan')
        core.steering.delta_hat = float('nan')
        cmd = core._condition(0.1, 1.0, DT)
        self.assertTrue(math.isfinite(cmd.steering_angle) and math.isfinite(cmd.speed))
        coasted = core._coast(DT)
        self.assertTrue(math.isfinite(coasted.steering_angle) and math.isfinite(coasted.speed))

    def test_coasting_is_logged_once_per_episode(self):
        class Log:
            def __init__(self):
                self.warnings = []

            def info(self, _m):
                pass

            def warn(self, m):
                self.warnings.append(m)

            def error(self, _m):
                pass

        log = Log()
        core = driver_core.DriverCore(cfg=driver_core.DriverConfig(),
                                      mpc_controller=_ScriptedMPC(), logger=log)
        core._coast(DT)
        core._coast(DT)
        self.assertEqual(len(log.warnings), 1)
        core._condition(0.0, 0.0, DT)               # recovered ...
        core._coast(DT)                             # ... and lost it again
        self.assertEqual(len(log.warnings), 2)


# ----------------------------------------------------------------------
# Continuous reactive speed cap
# ----------------------------------------------------------------------
class SpeedCapTest(unittest.TestCase):

    def _cap(self, ranges, steer=0.0):
        core, _ = _core()
        return core._speed_cap(np.asarray(ranges, dtype=float), ANGLES, steer)

    def test_the_cap_is_the_braking_distance_bound(self):
        expected = math.sqrt(2.0 * CFG.a_brake * (2.0 - CFG.stop_margin))
        self.assertAlmostEqual(self._cap(np.full(181, 2.0)), expected, places=9)

    def test_a_configured_stop_margin_is_kept_in_hand(self):
        core = driver_core.DriverCore(
            cfg=driver_core.DriverConfig(stop_margin=0.3), mpc_controller=_ScriptedMPC())
        cap = core._speed_cap(np.full(181, 2.0), ANGLES, 0.0)
        self.assertAlmostEqual(cap, math.sqrt(2.0 * CFG.a_brake * (2.0 - 0.3)), places=9)

    def test_the_cap_follows_the_low_end_of_the_front_sector_not_its_mean(self):
        ranges = np.full(181, 8.0)
        front = np.where(np.abs(ANGLES) <= 0.2)[0]
        ranges[front[:5]] = 2.0                     # ~22 % near; the mean is ~6.7 m
        expected = math.sqrt(2.0 * CFG.a_brake * (2.0 - CFG.stop_margin))
        self.assertAlmostEqual(self._cap(ranges), expected, places=9)

    def test_steering_derates_the_cap(self):
        expected = CFG.v_max / (1.0 + CFG.k_steer_derate * 0.4)
        self.assertAlmostEqual(self._cap(np.full(181, 8.0), steer=0.4), expected, places=9)
        self.assertAlmostEqual(self._cap(np.full(181, 8.0), steer=-0.4), expected, places=9)

    def test_the_cap_never_asks_for_less_than_crawl_or_more_than_v_max(self):
        self.assertEqual(self._cap(np.zeros(181)), CFG.crawl_speed)
        self.assertEqual(self._cap(np.full(181, 8.0)), CFG.v_max)

    def test_the_cap_is_continuous_in_clearance(self):
        caps = [self._cap(np.full(181, d)) for d in np.arange(0.0, 8.0, 0.01)]
        self.assertLess(float(np.max(np.abs(np.diff(caps)))), 0.01)

    def test_the_planned_speed_is_held_under_the_cap(self):
        core, _ = _core(accel=9.5)
        cmd = _drive(core, 1, v=1.7, ranges=np.full(181, 2.0))
        expected = math.sqrt(2.0 * CFG.a_brake * (2.0 - CFG.stop_margin))
        self.assertAlmostEqual(cmd.speed, expected, places=6)

    def test_a_cap_that_drops_is_obeyed_at_once_not_at_the_braking_rate(self):
        # A wall appears 1.5 m ahead of a car at v_max. The rate limit alone
        # would let the setpoint fall 0.02 m/s per step; the cap is a safety
        # ceiling and must not be delayed by it.
        core, _ = _core()
        cmd = _drive(core, 1, v=3.0, ranges=np.full(181, 1.5))
        expected = math.sqrt(2.0 * CFG.a_brake * (1.5 - CFG.stop_margin))
        self.assertAlmostEqual(cmd.speed, expected, places=6)

    def test_the_condition_step_clamps_to_a_cap_the_rate_limit_would_allow_past(self):
        core, _ = _core()
        core.speed = 2.0
        cmd = core._condition(0.0, 3.0, DT, speed_cap=1.2)
        self.assertEqual(cmd.speed, 1.2)

    def test_the_next_command_steps_up_from_the_capped_speed(self):
        core, _ = _core()
        core.speed = 2.0
        core._condition(0.0, 3.0, DT, speed_cap=1.2)
        cmd = core._condition(0.0, 3.0, DT)
        self.assertAlmostEqual(cmd.speed, 1.2 + CFG.accel_limit * DT, places=12)


# ----------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------
class ConditioningStatsTest(unittest.TestCase):

    def test_binds_are_counted_against_conditioned_steps(self):
        core, _ = _core()
        core._condition(0.0, 1.0, DT)
        core._condition(0.01, 1.0, DT)
        core._condition(0.02, 1.0, DT)
        core._condition(0.5, 1.0, DT)               # the servo could not follow this
        stats = core.conditioning_stats()
        self.assertEqual((stats.steps, stats.slew_binds, stats.coast_count), (4, 1, 0))
        self.assertAlmostEqual(stats.slew_bind_rate, 0.25)

    def test_clipping_to_the_steering_limit_is_not_a_slew_bind(self):
        core, _ = _core()
        for _ in range(20):
            core._condition(0.9, 1.0, DT)
        binds_while_ramping = core.conditioning_stats().slew_binds
        for _ in range(10):
            core._condition(0.9, 1.0, DT)
        self.assertEqual(core.conditioning_stats().slew_binds, binds_while_ramping)

    def test_an_idle_core_reports_a_zero_rate(self):
        core, _ = _core()
        self.assertEqual(core.conditioning_stats().slew_bind_rate, 0.0)


# ----------------------------------------------------------------------
# The harness exercises the same path
# ----------------------------------------------------------------------
def _result(**overrides):
    base = {'laps': 10, 'laps_requested': 10, 'collisions': 0, 'sat_duty': 0.0,
            'cmd_slew_mean': 0.005}
    base.update(overrides)
    return base


class HarnessVerdictTest(unittest.TestCase):

    def test_a_result_without_conditioning_metrics_still_passes(self):
        self.assertEqual(lap_sim.verdict(_result()), [])

    def test_a_limiter_that_binds_often_fails_the_verdict(self):
        reasons = lap_sim.verdict(_result(slew_bind_rate=lap_sim.MAX_SLEW_BIND_RATE * 2,
                                          coast_count=0))
        self.assertEqual(len(reasons), 1)
        self.assertIn('slew', reasons[0])

    def test_any_coast_entry_fails_the_verdict(self):
        reasons = lap_sim.verdict(_result(slew_bind_rate=0.0, coast_count=3))
        self.assertEqual(len(reasons), 1)
        self.assertIn('coast', reasons[0])

    def test_a_quiet_limiter_and_no_coasting_pass(self):
        self.assertEqual(lap_sim.verdict(_result(slew_bind_rate=0.0, coast_count=0)), [])


class HarnessMetricsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = lap_sim.HarnessConfig(laps=1, max_time=2.0, stale_steps=10,
                                    fixed_solve_ms=35.0, quiet=True)
        cls.result = lap_sim.run(cfg)

    def test_the_run_reports_the_conditioning_metrics(self):
        for key in ('slew_bind_rate', 'coast_count', 'cmd_speed_step_max'):
            self.assertIn(key, self.result)

    def test_a_clean_start_never_coasts(self):
        self.assertEqual(self.result['coast_count'], 0)

    def test_published_speed_steps_stay_inside_the_rate_limit(self):
        longest_interval = driver_core.DT_CLAMP_HIGH * DT
        bound = max(CFG.accel_limit, CFG.a_brake) * longest_interval + TOL
        self.assertLessEqual(self.result['cmd_speed_step_max'], bound)

    def test_a_run_that_takes_no_steps_reports_zeros(self):
        empty = lap_sim.run(lap_sim.HarnessConfig(laps=1, max_time=0.0, quiet=True))
        self.assertEqual(empty['slew_bind_rate'], 0.0)
        self.assertEqual(empty['coast_count'], 0)
        self.assertEqual(empty['cmd_speed_step_max'], 0.0)


if __name__ == '__main__':
    unittest.main()
