"""Tests for the Step 2 harness: timing model, lap counting and metrics."""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'team_driver'))

import lap_sim  # noqa: E402
import plant_sim  # noqa: E402


class HoldStepsTest(unittest.TestCase):

    def test_solve_time_scaled_by_rtf_sets_physics_steps_held(self):
        # 35 ms of wall-clock solve at RTF 1.4 is 49 ms of sim time.
        self.assertEqual(lap_sim.hold_steps(0.035, rtf=1.4, dt=0.01), 5)

    def test_a_fast_solve_still_holds_for_one_step(self):
        self.assertEqual(lap_sim.hold_steps(0.0001, rtf=1.4, dt=0.01), 1)

    def test_measured_solve_is_scaled_to_container_speed(self):
        cfg = lap_sim.HarnessConfig(rtf=1.4, solve_scale=2.0)
        # 25 ms offline -> 50 ms in the container -> 70 ms of sim -> 7 steps.
        self.assertEqual(lap_sim.hold_for(cfg, 0.025), 7)

    def test_fixed_solve_time_ignores_the_scale(self):
        cfg = lap_sim.HarnessConfig(rtf=1.4, solve_scale=2.0, fixed_solve_ms=35.0)
        self.assertEqual(lap_sim.hold_for(cfg, 0.200), 5)


class SnapshotRingTest(unittest.TestCase):

    def test_returns_snapshot_k_steps_before_newest(self):
        ring = lap_sim.SnapshotRing(stale_steps=3, initial=('s0', 0.0))
        for i in range(1, 8):
            ring.push((f's{i}', i * 0.01))
        self.assertEqual(ring.stale()[0], 's4')

    def test_zero_staleness_returns_newest(self):
        ring = lap_sim.SnapshotRing(stale_steps=0, initial=('s0', 0.0))
        ring.push(('s1', 0.01))
        self.assertEqual(ring.stale()[0], 's1')

    def test_before_the_ring_fills_it_returns_the_oldest_available(self):
        ring = lap_sim.SnapshotRing(stale_steps=10, initial=('s0', 0.0))
        ring.push(('s1', 0.01))
        self.assertEqual(ring.stale()[0], 's0')

    def test_negative_staleness_is_rejected(self):
        with self.assertRaises(ValueError):
            lap_sim.SnapshotRing(stale_steps=-1, initial=('s0', 0.0))


class FinishLineTest(unittest.TestCase):

    def setUp(self):
        self.line = lap_sim.FinishLine(x=1.32, y_min=-0.76, y_max=0.89)

    def test_first_crossing_is_the_out_lap_and_scores_no_lap(self):
        self.assertFalse(self.line.update((1.0, 0.0), (1.5, 0.0), 2.0))
        self.assertEqual(self.line.laps, 0)

    def test_second_crossing_scores_a_timed_lap(self):
        self.line.update((1.0, 0.0), (1.5, 0.0), 2.0)
        self.assertTrue(self.line.update((1.0, 0.0), (1.5, 0.0), 55.0))
        self.assertEqual(self.line.laps, 1)
        self.assertAlmostEqual(self.line.lap_times[0], 53.0)

    def test_backwards_crossing_does_not_count(self):
        self.line.update((1.0, 0.0), (1.5, 0.0), 2.0)
        self.assertFalse(self.line.update((1.5, 0.0), (1.0, 0.0), 3.0))
        self.assertFalse(self.line.update((1.0, 0.0), (1.5, 0.0), 4.0))
        self.assertEqual(self.line.laps, 0)

    def test_crossing_outside_the_line_span_does_not_count(self):
        self.assertFalse(self.line.update((1.0, 5.0), (1.5, 5.0), 2.0))
        self.assertEqual(self.line.crossings, 0)


class ChatterMetricsTest(unittest.TestCase):

    def test_reversals_ignore_repeated_commands(self):
        steers = np.array([0.0, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2])
        self.assertEqual(lap_sim.count_reversals(steers), 2)

    def test_monotone_sequence_has_no_reversals(self):
        self.assertEqual(lap_sim.count_reversals(np.linspace(0, 1, 10)), 0)


class InstantSteerPlantTest(unittest.TestCase):

    def test_steering_reaches_the_command_in_one_step(self):
        plant = lap_sim.InstantSteerPlant()
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.3, 1.0)
        plant.step()
        self.assertAlmostEqual(float(plant.state[plant_sim.DELTA]), 0.3, places=5)
        self.assertFalse(plant.is_rate_saturated)


class PlantChoiceTest(unittest.TestCase):

    def test_each_plant_flag_builds_its_plant(self):
        self.assertIs(type(lap_sim.make_plant('st')), plant_sim.GymPlant)
        self.assertIs(type(lap_sim.make_plant('kinematic')), plant_sim.KinematicPlant)
        self.assertIs(type(lap_sim.make_plant('instant')), lap_sim.InstantSteerPlant)

    def test_unknown_plant_is_rejected(self):
        with self.assertRaises(ValueError):
            lap_sim.make_plant('bogus')


class CommandLineTest(unittest.TestCase):

    def test_defaults_are_the_calibrated_judged_timing(self):
        cfg = lap_sim.parse_config([])
        # Measured live (probe run 20260930T091032Z): scans are 9 steps old,
        # and a ~32 ms offline solve becomes a ~74 ms hold in the container.
        self.assertEqual(cfg.stale_steps, 9)
        self.assertEqual(cfg.rtf, 1.4)
        self.assertEqual(cfg.solve_scale, 1.65)
        self.assertEqual(cfg.plant, 'st')
        self.assertIsNone(cfg.seed)

    def test_every_plan_flag_is_accepted(self):
        cfg = lap_sim.parse_config(
            ['--laps', '10', '--stale-steps', '20', '--rtf', '1.3', '--plant', 'kinematic',
             '--fixed-solve-ms', '35', '--solve-scale', '1.5', '--seed', '7'])
        self.assertEqual((cfg.laps, cfg.stale_steps, cfg.rtf, cfg.plant),
                         (10, 20, 1.3, 'kinematic'))
        self.assertEqual((cfg.fixed_solve_ms, cfg.solve_scale, cfg.seed), (35.0, 1.5, 7))


def _result(**overrides):
    base = dict(laps=10, laps_requested=10, collisions=0, sat_duty=0.0,
                cmd_slew_mean=0.005)
    base.update(overrides)
    return base


class VerdictTest(unittest.TestCase):

    def test_a_clean_run_passes(self):
        self.assertEqual(lap_sim.verdict(_result()), [])

    def test_each_failure_is_reported(self):
        reasons = lap_sim.verdict(_result(laps=3, collisions=2, sat_duty=0.2,
                                          cmd_slew_mean=0.05))
        self.assertEqual(len(reasons), 4)


class HarnessGateTest(unittest.TestCase):
    """Step 2's binding gate: the harness must reproduce the judged failure."""

    def test_judged_like_failure_is_reproduced(self):
        self.assertTrue(lap_sim.reproduces_judged_failure(
            _result(collisions=3, sat_duty=0.21)))

    def test_collisions_are_scaled_to_ten_laps(self):
        # 2 collisions in 5 laps is 4 per 10 laps.
        self.assertTrue(lap_sim.reproduces_judged_failure(
            _result(laps=5, collisions=2, sat_duty=0.21)))

    def test_too_few_collisions_is_not_a_reproduction(self):
        self.assertFalse(lap_sim.reproduces_judged_failure(
            _result(collisions=2, sat_duty=0.207)))

    def test_saturation_just_under_the_live_value_is_a_reproduction(self):
        # Live duty is 20.7 %; a faithful harness lands either side of it.
        self.assertTrue(lap_sim.reproduces_judged_failure(
            _result(collisions=7, sat_duty=0.19)))

    def test_saturation_well_below_live_is_not_a_reproduction(self):
        self.assertFalse(lap_sim.reproduces_judged_failure(
            _result(collisions=7, sat_duty=0.17)))

    def test_saturation_well_above_live_is_not_a_reproduction(self):
        self.assertFalse(lap_sim.reproduces_judged_failure(
            _result(collisions=7, sat_duty=0.25)))

    def test_no_laps_counts_as_a_reproduction_only_with_collisions(self):
        self.assertFalse(lap_sim.reproduces_judged_failure(
            _result(laps=0, collisions=0, sat_duty=0.207)))


class KinematicClosedLoopTest(unittest.TestCase):

    def test_the_kinematic_ablation_drives_and_records_its_seed(self):
        cfg = lap_sim.HarnessConfig(laps=1, max_time=2.0, fixed_solve_ms=35.0,
                                    plant='kinematic', seed=3, quiet=True)
        result = lap_sim.run(cfg)
        self.assertGreater(result['distance_m'], 0.5)
        self.assertEqual(result['plant'], 'kinematic')
        self.assertEqual(result['seed'], 3)


class _RecordingPlant(plant_sim.GymPlant):
    """Logs the command in force on every physics step."""

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.speed_cmds = []
        self.steer_cmds = []

    def step(self):
        self.speed_cmds.append(self.speed_cmd)
        self.steer_cmds.append(self.steer_cmd)
        return super().step()


def _run_on(plant, **cfg):
    with mock.patch.object(lap_sim, 'make_plant', lambda name, seed=None: plant):
        return lap_sim.run(lap_sim.HarnessConfig(quiet=True, **cfg))


class CommandTimingTest(unittest.TestCase):
    """A command reaches the plant only after the solve that produced it."""

    def test_previous_command_stays_latched_while_the_next_one_solves(self):
        plant = _RecordingPlant()
        _run_on(plant, laps=1, max_time=0.2, stale_steps=0, fixed_solve_ms=35.0)
        # 35 ms at RTF 1.4 is 5 steps: the first 5 run on the initial (zero)
        # command, and the first real command lands on step 6.
        self.assertEqual(plant.speed_cmds[:5], [0.0] * 5)
        self.assertGreater(plant.speed_cmds[5], 0.0)

    def test_every_command_is_held_for_exactly_one_solve(self):
        plant = _RecordingPlant()
        _run_on(plant, laps=1, max_time=0.3, stale_steps=0, fixed_solve_ms=35.0)
        pairs = list(zip(plant.steer_cmds, plant.speed_cmds))
        changes = [i for i in range(1, len(pairs)) if pairs[i] != pairs[i - 1]]
        self.assertTrue(changes, 'no command ever reached the plant')
        self.assertTrue(all(i % 5 == 0 for i in changes), changes)

    def test_a_disqualification_stops_the_run_mid_hold(self):
        class DisqualifyAtOnce(lap_sim.scan_sim.CollisionCounter):
            """Disqualifies on its first update, mid-way through a hold."""

            def update(self, time_s, in_contact):
                self.updated = True
                return super().update(time_s, in_contact)

            @property
            def disqualified(self):
                return getattr(self, 'updated', False)

        plant = _RecordingPlant()
        with mock.patch.object(lap_sim.scan_sim, 'CollisionCounter', DisqualifyAtOnce):
            _run_on(plant, laps=1, max_time=1.0, stale_steps=0, fixed_solve_ms=35.0)
        self.assertEqual(len(plant.speed_cmds), 1)


class SolveClockTest(unittest.TestCase):
    """Solve time is the solver's own CPU work, not wall time on a busy host."""

    def test_time_the_solver_spends_descheduled_is_not_counted(self):
        import time
        real_step = lap_sim.DriverCore.step

        def descheduled_step(core, t_sim=None):
            time.sleep(0.06)            # wall time passes, no CPU is used
            return real_step(core, t_sim=t_sim)

        cfg = lap_sim.HarnessConfig(laps=1, max_time=0.3, fixed_solve_ms=35.0, quiet=True)
        baseline = lap_sim.run(cfg)['solve_ms_mean']
        with mock.patch.object(lap_sim.DriverCore, 'step', descheduled_step):
            slept = lap_sim.run(cfg)['solve_ms_mean']
        # A wall clock would add the full 60 ms of sleep to every solve.
        self.assertLess(slept, baseline + 30.0)


class EmptyRunTest(unittest.TestCase):

    def test_a_run_that_takes_no_steps_still_summarises(self):
        result = lap_sim.run(lap_sim.HarnessConfig(laps=1, max_time=0.0, quiet=True))
        self.assertEqual(result['distance_m'], 0.0)
        self.assertEqual(result['hold_steps_max'], 0)
        self.assertEqual(result['sat_duty'], 0.0)


class ClosedLoopSmokeTest(unittest.TestCase):
    """A short run through the real DriverCore, plant, scanner and contact."""

    @classmethod
    def setUpClass(cls):
        cfg = lap_sim.HarnessConfig(laps=1, max_time=2.0, stale_steps=10,
                                    fixed_solve_ms=35.0, quiet=True)
        cls.result = lap_sim.run(cfg)
        cls.result_again = lap_sim.run(cfg)

    def test_the_car_moves_off_the_start(self):
        self.assertGreater(self.result['distance_m'], 0.5)

    def test_holds_match_the_fixed_solve_time(self):
        # 35 ms at RTF 1.4 -> 5 physics steps per command.
        self.assertEqual(self.result['hold_steps_mean'], 5.0)
        self.assertAlmostEqual(self.result['dead_time_ms_mean'], 150.0)

    def test_fixed_solve_time_makes_the_run_deterministic(self):
        for key in ('distance_m', 'sat_duty', 'cmd_slew_mean', 'collisions'):
            self.assertEqual(self.result[key], self.result_again[key], key)

    def test_reports_every_gate_metric(self):
        for key in ('sat_duty', 'cmd_slew_mean', 'actual_slew_mean',
                    'track_err_mean', 'track_err_p95', 'collisions',
                    'reversals_per_m', 'laps', 'lap_times'):
            self.assertIn(key, self.result)


if __name__ == '__main__':
    unittest.main()
