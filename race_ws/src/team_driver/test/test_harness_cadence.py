"""The harness runs the control loop at the node's fixed 25 ms timer.

Since Step 3 the live node plans on a `create_timer(control_period)`, not once
per scan. `lap_sim.py` still advanced the plant by a solve-based hold, so with a
fast solve it ran the controller at ~100 Hz (1 physics step per command) and
every harness sweep tested a faster controller than the one judged. The plant
grid is 10 ms, so a 25 ms period is 3 and 2 steps alternately.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'team_driver'))

import lap_sim  # noqa: E402

PERIOD = 0.025
DT = lap_sim.PHYSICS_DT


class CadenceStepsTest(unittest.TestCase):

    def test_a_25_ms_period_alternates_three_and_two_steps(self):
        steps = [lap_sim.cadence_steps(k, PERIOD, DT) for k in range(8)]
        self.assertEqual(steps, [3, 2, 3, 2, 3, 2, 3, 2])

    def test_the_steps_add_up_to_the_period_with_no_drift(self):
        total = sum(lap_sim.cadence_steps(k, PERIOD, DT) for k in range(4000))
        self.assertEqual(total, 10000)   # 4000 x 25 ms = 100 s = 10000 steps

    def test_a_period_on_the_grid_is_constant(self):
        self.assertEqual({lap_sim.cadence_steps(k, 0.02, DT) for k in range(50)}, {2})
        self.assertEqual({lap_sim.cadence_steps(k, 0.03, DT) for k in range(50)}, {3})

    def test_a_period_shorter_than_a_step_still_advances_one_step(self):
        self.assertEqual({lap_sim.cadence_steps(k, 0.004, DT) for k in range(50)}, {1})


class PeriodForTest(unittest.TestCase):

    def test_a_fast_solve_leaves_the_period_to_the_timer(self):
        cfg = lap_sim.HarnessConfig()
        self.assertEqual([lap_sim.period_for(cfg, k, latency=1) for k in range(4)],
                         [3, 2, 3, 2])

    def test_a_solve_longer_than_the_period_stretches_it(self):
        # A real timer overruns: the next tick fires as soon as the callback ends.
        cfg = lap_sim.HarnessConfig()
        self.assertEqual({lap_sim.period_for(cfg, k, latency=5) for k in range(4)}, {5})

    def test_a_latency_between_the_two_cadence_steps_binds_only_on_short_ticks(self):
        # 2 steps of latency fits the 3-step ticks and exactly fills the 2-step
        # ones; 3 steps stretches the 2-step ticks to 3.
        cfg = lap_sim.HarnessConfig()
        self.assertEqual([lap_sim.period_for(cfg, k, latency=2) for k in range(4)],
                         [3, 2, 3, 2])
        self.assertEqual([lap_sim.period_for(cfg, k, latency=3) for k in range(4)],
                         [3, 3, 3, 3])

    def test_a_zero_control_period_is_the_old_solve_based_hold(self):
        cfg = lap_sim.HarnessConfig(control_period=0.0)
        self.assertEqual({lap_sim.period_for(cfg, k, latency=1) for k in range(4)}, {1})
        self.assertEqual({lap_sim.period_for(cfg, k, latency=5) for k in range(4)}, {5})


class ConfigTest(unittest.TestCase):

    def test_the_default_period_is_the_nodes_control_period(self):
        self.assertEqual(lap_sim.HarnessConfig().control_period, 0.025)
        self.assertEqual(lap_sim.DEFAULT_CONTROL_PERIOD,
                         lap_sim.JUDGED_MPC['dt'])

    def test_the_flag_sets_it(self):
        self.assertEqual(lap_sim.parse_config([]).control_period, 0.025)
        self.assertEqual(lap_sim.parse_config(['--control-period', '0']).control_period, 0.0)
        self.assertEqual(lap_sim.parse_config(['--control-period', '0.05']).control_period, 0.05)


class _FakePlant:
    """Just enough plant for `_advance`: a clock that ticks 10 ms per step."""

    is_rate_saturated = False
    steer_cmd = 0.0
    state = [0.0] * 7
    base_link_pose = (0.0, 0.0, 0.0)

    def __init__(self):
        self.time = 0.0
        self.steps = 0

    def step(self):
        self.steps += 1
        self.time += DT


class _FakeCounter:
    def __init__(self, disqualify_after):
        self.disqualify_after = disqualify_after
        self.updates = 0

    @property
    def disqualified(self):
        return self.updates >= self.disqualify_after

    def update(self, *_):
        self.updates += 1


class _FakeLine:
    laps = 0

    def update(self, *_):
        return False


class _FakeRing:
    def push(self, _snapshot):
        pass


class _FakeContact:
    def resolve(self, _state):
        return ()


class AdvanceTest(unittest.TestCase):

    def _advance(self, steps, disqualify_after):
        plant = _FakePlant()
        cfg = lap_sim.HarnessConfig(laps=1, quiet=True)
        result = lap_sim._advance(steps, plant, _FakeContact(), _FakeCounter(disqualify_after),
                                  _FakeLine(), _FakeRing(), lap_sim._Log(), (0.0, 0.0), cfg)
        return plant, result

    def test_it_runs_every_step_when_nothing_ends_the_run(self):
        plant, (finished, _) = self._advance(3, disqualify_after=99)
        self.assertEqual((plant.steps, finished), (3, False))

    def test_it_stops_at_the_step_that_ends_the_run(self):
        plant, (finished, _) = self._advance(5, disqualify_after=2)
        self.assertEqual((plant.steps, finished), (2, True))

    def test_zero_steps_is_a_no_op(self):
        plant, (finished, xy) = self._advance(0, disqualify_after=0)
        self.assertEqual((plant.steps, finished, xy), (0, False, (0.0, 0.0)))


def _latch_times(**overrides):
    """Plant time of every command latch in a short run, and the run's result."""
    times = []
    real_make = lap_sim.make_plant

    def make(*args, **kwargs):
        plant = real_make(*args, **kwargs)
        real_latch = plant.latch

        def latch(*a, **k):
            times.append(plant.time)
            return real_latch(*a, **k)

        plant.latch = latch
        return plant

    cfg = lap_sim.HarnessConfig(laps=1, max_time=2.0, stale_steps=9, quiet=True, **overrides)
    with mock.patch.object(lap_sim, 'make_plant', make):
        result = lap_sim.run(cfg)
    return times, result


class ClosedLoopCadenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fast_times, cls.fast = _latch_times(fixed_solve_ms=7.0)
        cls.slow_times, cls.slow = _latch_times(fixed_solve_ms=35.0)
        cls.legacy_times, cls.legacy = _latch_times(fixed_solve_ms=7.0, control_period=0.0)

    def test_commands_are_issued_every_25_ms_regardless_of_the_solve(self):
        times = self.fast_times
        gaps = [round(b - a, 6) for a, b in zip(times, times[1:])]
        self.assertEqual(set(gaps), {0.02, 0.03})
        self.assertAlmostEqual((times[-1] - times[0]) / (len(times) - 1), PERIOD, places=3)

    def test_a_command_latches_one_solve_after_the_tick(self):
        # 7 ms at RTF 1.4 is one 10 ms step: the first tick is at t = 0.
        self.assertAlmostEqual(self.fast_times[0], 0.01, places=6)

    def test_the_hold_stats_describe_the_period(self):
        self.assertAlmostEqual(self.fast['hold_steps_mean'], 2.5, delta=0.02)
        self.assertEqual(self.fast['hold_steps_max'], 3)

    def test_dead_time_counts_the_solve_latency_not_the_period(self):
        # 9 stale steps + 1 latency step = 100 ms, however long the period.
        self.assertAlmostEqual(self.fast['latency_steps_mean'], 1.0)
        self.assertAlmostEqual(self.fast['dead_time_ms_mean'], 100.0)

    def test_an_overrunning_solve_still_sets_the_period(self):
        self.assertEqual(self.slow['hold_steps_mean'], 5.0)
        self.assertAlmostEqual(self.slow['dead_time_ms_mean'], 140.0)   # 9 stale + 5

    def test_the_old_behaviour_is_still_available(self):
        self.assertEqual(self.legacy['hold_steps_mean'], 1.0)
        gaps = {round(b - a, 6) for a, b in zip(self.legacy_times, self.legacy_times[1:])}
        self.assertEqual(gaps, {0.01})


if __name__ == '__main__':
    unittest.main()
