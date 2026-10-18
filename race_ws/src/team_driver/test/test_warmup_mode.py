"""Sub-9 phase 2 Step 1: a cautious out lap and warm-up lap, the race profile after.

Neither the out lap nor the warm-up lap is timed, but a collision in either still
counts toward the disqualification limit. Judged warm-ups hit in 3 of 13 sub-9 runs
(0 in 130 timed laps), at scattered spots after early host stalls.
Until the car has crossed the start line `warmup_crossings` times and then reached
`warmup_end_s`, the profile takes extra sectors and the acceleration is scaled.
Off (0 crossings) must be the race car exactly; the default is on (1 crossing).
"""

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
import driver_params  # noqa: E402
import lap_sim  # noqa: E402
import speed_profile as sp  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

WARM = dict(warmup_crossings=1, warmup_end_s=60.0,
            warmup_sector_scale=(0.0, 20.0, 0.8), warmup_accel_scale=0.6)


def _core(**cfg):
    mpc = MPCController(vehicle_model=VehicleModel(), **lap_sim.JUDGED_MPC)
    return driver_core.DriverCore(cfg=driver_core.DriverConfig(**cfg), mpc_controller=mpc)


def _arc(path):
    seg = np.linalg.norm(np.diff(np.asarray(path)[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


OFF = dict(warmup_crossings=0)


class OffIsTheSub9CarTest(unittest.TestCase):
    def test_the_pinned_default_is_a_slow_warm_up_to_the_last_corner(self):
        cfg = driver_core.DriverConfig()
        self.assertEqual((cfg.warmup_crossings, cfg.warmup_end_s, cfg.warmup_sector_scale,
                          cfg.warmup_accel_scale), (1, 56.0, (0.0, 56.0, 0.35), 0.5))

    def test_off_tracks_the_race_profile_from_the_start(self):
        core = _core(**OFF)
        cfg = core.cfg
        expected = sp.speed_profile(
            core.reference_path, a_lat_max=cfg.a_lat_max, lock_margin=cfg.lock_margin,
            a_accel_max=cfg.a_accel_max, a_brake_max=cfg.a_brake_max, v_max=cfg.v_max,
            sector_scale=sp.sector_triples(cfg.sector_scale),
            gg_brake=cfg.gg_brake, gg_floor=cfg.gg_floor, gg_exponent=cfg.gg_exponent,
            curvature_smooth=cfg.curvature_smooth)
        np.testing.assert_array_equal(core.reference_speeds, expected)
        np.testing.assert_array_equal(core.race_speeds, expected)
        self.assertFalse(core.warming_up)

    def test_off_leaves_the_acceleration_bounds_alone(self):
        core = _core(**OFF)
        self.assertEqual(core.accel_limit, core.cfg.accel_limit)
        p = core.mpc_controller.vehicle_model.params
        self.assertEqual(core.mpc_controller.control_max[1], p.a_max)


class WarmProfileTest(unittest.TestCase):
    def setUp(self):
        self.core = _core(**WARM)

    def test_it_starts_warming_up(self):
        self.assertTrue(self.core.warming_up)

    def test_the_warm_profile_never_exceeds_the_race_profile(self):
        warm, race = self.core.reference_speeds, self.core.race_speeds
        self.assertTrue(np.all(warm <= race))

    def test_the_warm_sector_is_slower(self):
        s = _arc(self.core.reference_path)
        inside = (s >= 5.0) & (s < 15.0)
        self.assertTrue(np.any(self.core.reference_speeds[inside]
                               < self.core.race_speeds[inside] - 0.1))

    def test_the_acceleration_is_scaled_while_warming_up(self):
        core = self.core
        p = core.mpc_controller.vehicle_model.params
        self.assertAlmostEqual(core.accel_limit, core.cfg.accel_limit * 0.6)
        self.assertAlmostEqual(core.mpc_controller.control_max[1], p.a_max * 0.6)
        self.assertIs(core.mpc_controller.reference_speeds, core.reference_speeds)


class CrossingTest(unittest.TestCase):
    def _feed(self, core, arcs):
        # A solve moves the arc ~0.3 m; only the line makes it jump over half a lap.
        for arc in arcs:
            core._note_arc(arc)

    def test_the_out_lap_past_the_end_arc_does_not_end_it(self):
        core = _core(**WARM)
        self._feed(core, [71.0, 72.5, 74.0])
        self.assertTrue(core.warming_up)

    def test_it_ends_at_the_end_arc_after_the_crossings(self):
        core = _core(**WARM)
        self._feed(core, [71.0, 74.0, 0.2, 10.0, 30.0, 50.0, 59.9])
        self.assertTrue(core.warming_up)
        core._note_arc(60.1)
        self.assertFalse(core.warming_up)

    def test_after_it_ends_the_race_profile_and_bounds_are_back(self):
        core = _core(**WARM)
        self._feed(core, [71.0, 74.0, 0.2, 30.0, 60.1])
        np.testing.assert_array_equal(core.reference_speeds, core.race_speeds)
        self.assertIs(core.mpc_controller.reference_speeds, core.race_speeds)
        p = core.mpc_controller.vehicle_model.params
        self.assertEqual(core.accel_limit, core.cfg.accel_limit)
        self.assertEqual(core.mpc_controller.control_max[1], p.a_max)

    def test_jitter_across_the_line_counts_once(self):
        core = _core(**{**WARM, 'warmup_crossings': 2})
        self._feed(core, [73.9, 0.1, 74.2, 0.3, 30.0, 61.0])
        self.assertTrue(core.warming_up)        # one crossing, not two
        self._feed(core, [74.0, 0.2, 30.0, 60.5])
        self.assertFalse(core.warming_up)

    def test_unsolved_cycles_are_ignored(self):
        core = _core(**WARM)
        self._feed(core, [71.0, None, float('nan'), 74.0, 0.2, None, 30.0, 60.1])
        self.assertFalse(core.warming_up)

    def test_a_sim_reset_starts_the_warm_up_again(self):
        core = _core(**WARM)
        self._feed(core, [71.0, 74.0, 0.2, 30.0, 60.1])
        core._restart_warmup()
        self.assertTrue(core.warming_up)
        self.assertTrue(np.all(core.reference_speeds <= core.race_speeds))
        self.assertLess(core.accel_limit, core.cfg.accel_limit)

    def test_the_mpcs_unwrapped_arc_counts_its_crossings(self):
        # MPCController.arc_position keeps counting past the lap length (71 -> 148 m
        # over the out lap and warm-up lap); only its position within the lap wraps.
        core = _core(**{**WARM, 'warmup_crossings': 2})
        L = core._reference.length
        self._feed(core, [71.0, L - 0.2, L + 0.1, L + 30.0, L + 60.5])
        self.assertTrue(core.warming_up)        # one crossing so far
        self._feed(core, [2 * L - 0.1, 2 * L + 0.2, 2 * L + 30.0, 2 * L + 59.9])
        self.assertTrue(core.warming_up)        # two crossings, not yet at s 60
        core._note_arc(2 * L + 60.1)
        self.assertFalse(core.warming_up)

    def test_off_ignores_the_arc(self):
        core = _core(**OFF)
        self._feed(core, [71.0, 74.0, 0.2, 30.0, 60.1])
        self.assertFalse(core.warming_up)


class ValidationTest(unittest.TestCase):
    def _values(self, **over):
        values = dict(driver_params.DEFAULTS)
        values.update(over)
        return values

    def test_the_defaults_pass(self):
        driver_params.validate(self._values())

    def test_bad_warm_up_values_are_refused(self):
        bad = [dict(warmup_crossings=-1), dict(warmup_crossings=1.5),
               dict(warmup_accel_scale=0.0), dict(warmup_accel_scale=1.2),
               dict(warmup_sector_scale=[0.0, 20.0, 1.1]),
               dict(warmup_sector_scale=[0.0, 20.0]),
               dict(warmup_end_s=math.nan), dict(warmup_end_s=-1.0)]
        for over in bad:
            with self.subTest(**{k: str(v) for k, v in over.items()}):
                with self.assertRaises(ValueError):
                    driver_params.validate(self._values(**over))
                with self.assertRaises(ValueError):
                    driver_core.DriverConfig(**{k: tuple(v) if isinstance(v, list) else v
                                                for k, v in over.items()})

    def test_the_harness_takes_an_integer_crossing_count(self):
        self.assertEqual(lap_sim.parse_driver_override('warmup_crossings=1'),
                         ('warmup_crossings', 1))
        self.assertIsInstance(lap_sim.parse_driver_override('warmup_crossings=2')[1], int)
        with self.assertRaises(Exception):
            lap_sim.parse_driver_override('warmup_crossings=1.5')

    def test_the_harness_takes_the_warm_sectors(self):
        self.assertEqual(lap_sim.parse_driver_override('warmup_sector_scale=0,20,0.8'),
                         ('warmup_sector_scale', (0.0, 20.0, 0.8)))


if __name__ == '__main__':
    unittest.main()
