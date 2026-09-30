"""Step 3: dt is measured from sensor stamps, and each scan is planned once."""

import unittest

import numpy as np

from team_driver.driver_core import DriverConfig, DriverCore, DtTracker

NOMINAL = 0.025


class _StubMPC:
    dt = NOMINAL

    def __init__(self, accel=2.0):
        self.accel = accel
        self.calls = 0

    def speed_profile(self, **_kw):
        return np.full(4, 3.0)

    def set_reference_path(self, *_a):
        pass

    def get_control(self, _state):
        self.calls += 1
        return np.array([0.0, self.accel]), {'solve_time_ms': 0.0}


def _core(accel=2.0):
    mpc = _StubMPC(accel)
    core = DriverCore(cfg=DriverConfig(), mpc_controller=mpc)
    core.reference_path = np.zeros((4, 2))
    core.set_odom(0.0, 0.0, 0.0, 1.0, t_sim=0.0)
    return core, mpc


def _scan(core, t):
    n = 181
    core.set_scan(np.full(n, 6.0), -np.pi / 2, np.pi / (n - 1), t_sim=t)


class DtTrackerTest(unittest.TestCase):

    def test_starts_at_nominal_before_any_stamp(self):
        self.assertEqual(DtTracker(NOMINAL).dt, NOMINAL)

    def test_first_stamp_gives_no_interval(self):
        tracker = DtTracker(NOMINAL)
        tracker.update(1.0)
        self.assertEqual(tracker.dt, NOMINAL)

    def test_dt_is_the_delta_between_consecutive_stamps(self):
        tracker = DtTracker(NOMINAL)
        tracker.update(1.0)
        tracker.update(1.030)
        self.assertAlmostEqual(tracker.dt, 0.030)

    def test_a_dropped_scan_is_clamped_to_three_times_nominal(self):
        tracker = DtTracker(NOMINAL)
        tracker.update(1.0)
        tracker.update(2.0)
        self.assertAlmostEqual(tracker.dt, 3.0 * NOMINAL)

    def test_a_tiny_interval_is_clamped_to_half_nominal(self):
        tracker = DtTracker(NOMINAL)
        tracker.update(1.0)
        tracker.update(1.001)
        self.assertAlmostEqual(tracker.dt, 0.5 * NOMINAL)

    def test_a_missing_stamp_keeps_the_previous_dt(self):
        tracker = DtTracker(NOMINAL)
        tracker.update(1.0)
        tracker.update(1.030)
        tracker.update(None)
        self.assertAlmostEqual(tracker.dt, 0.030)

    def test_a_stamp_that_goes_backwards_is_clamped_not_negative(self):
        tracker = DtTracker(NOMINAL)
        tracker.update(2.0)
        tracker.update(1.0)
        self.assertAlmostEqual(tracker.dt, 0.5 * NOMINAL)

    def test_summary_reports_mean_p95_and_max(self):
        tracker = DtTracker(NOMINAL)
        t = 0.0
        tracker.update(t)
        for _ in range(99):
            t += NOMINAL
            tracker.update(t)
        t += 0.050
        tracker.update(t)
        mean, p95, worst = tracker.summary()
        self.assertAlmostEqual(worst, 0.050)
        self.assertAlmostEqual(p95, NOMINAL)
        self.assertGreater(mean, NOMINAL)

    def test_summary_of_an_empty_tracker_is_nominal(self):
        self.assertEqual(DtTracker(NOMINAL).summary(), (NOMINAL,) * 3)

    def test_reset_stats_clears_history_but_keeps_dt(self):
        tracker = DtTracker(NOMINAL)
        tracker.update(1.0)
        tracker.update(1.030)
        tracker.reset_stats()
        self.assertEqual(tracker.summary(), (NOMINAL,) * 3)
        self.assertAlmostEqual(tracker.dt, 0.030)


class DriverCoreDeadTimeTest(unittest.TestCase):

    def test_speed_projection_uses_the_measured_dt(self):
        core, _ = _core(accel=2.0)
        _scan(core, 1.000)
        core.step(t_sim=1.000)
        _scan(core, 1.050)
        cmd = core.step(t_sim=1.050)
        # v=1.0, a=2.0, measured dt 0.050 -> 1.10, not 1.0 + 2 * 0.025
        self.assertAlmostEqual(cmd.speed, 1.0 + 2.0 * 0.050, places=6)

    def test_the_same_scan_stamp_is_not_planned_twice(self):
        core, mpc = _core()
        _scan(core, 1.000)
        first = core.step(t_sim=1.000)
        second = core.step(t_sim=1.010)
        self.assertEqual(mpc.calls, 1)
        self.assertEqual(second, first)

    def test_a_new_scan_stamp_is_planned(self):
        core, mpc = _core()
        _scan(core, 1.000)
        core.step(t_sim=1.000)
        _scan(core, 1.025)
        core.step(t_sim=1.025)
        self.assertEqual(mpc.calls, 2)

    def test_unstamped_scans_are_always_planned(self):
        core, mpc = _core()
        core.set_scan(np.full(181, 6.0), -np.pi / 2, np.pi / 180)
        core.step()
        core.step()
        self.assertEqual(mpc.calls, 2)

    def test_dt_measured_comes_from_scan_stamps_not_step_times(self):
        core, _ = _core()
        _scan(core, 1.000)
        _scan(core, 1.030)
        self.assertAlmostEqual(core.dt_measured, 0.030)


if __name__ == '__main__':
    unittest.main()
