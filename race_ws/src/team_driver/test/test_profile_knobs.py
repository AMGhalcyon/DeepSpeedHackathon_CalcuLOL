"""Sub-9 phase 2 Step 1: the profile's g-g exponent and curvature smoothing as driver fields.

Both passed the offline floor screen (exponent 3: 8.929 -> 8.813 s; smoothing 13:
-> 8.809 s). The module constants stay the old car; the phase-2 pin is 5 and 21.
"""

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
import raceline  # noqa: E402
import speed_profile as sp  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402


def _core(**cfg):
    mpc = MPCController(vehicle_model=VehicleModel(), **lap_sim.JUDGED_MPC)
    return driver_core.DriverCore(cfg=driver_core.DriverConfig(**cfg), mpc_controller=mpc)


def _expected(core, **kw):
    cfg = core.cfg
    return sp.speed_profile(
        core.reference_path, a_lat_max=cfg.a_lat_max, lock_margin=cfg.lock_margin,
        a_accel_max=cfg.a_accel_max, a_brake_max=cfg.a_brake_max, v_max=cfg.v_max,
        sector_scale=sp.sector_triples(cfg.sector_scale),
        gg_brake=cfg.gg_brake, gg_floor=cfg.gg_floor, **kw)


class KnobTest(unittest.TestCase):
    def test_the_module_constants_are_the_old_car_and_the_defaults_the_pin(self):
        self.assertEqual((sp.GG_EXPONENT, sp.CURVATURE_SMOOTH), (2.0, 9))
        cfg = driver_core.DriverConfig()
        self.assertEqual((cfg.gg_exponent, cfg.curvature_smooth), (5.0, 21))

    def test_the_driver_profiles_with_them(self):
        core = _core(gg_exponent=3.0, curvature_smooth=13)
        np.testing.assert_array_equal(
            core.race_speeds, _expected(core, gg_exponent=3.0, curvature_smooth=13))

    def test_bad_values_are_refused(self):
        for over in (dict(gg_exponent=0.0), dict(curvature_smooth=8),
                     dict(curvature_smooth=0), dict(curvature_smooth=9.0)):
            with self.subTest(**over):
                with self.assertRaises(ValueError):
                    driver_core.DriverConfig(**over)
                with self.assertRaises(ValueError):
                    driver_params.validate({**driver_params.DEFAULTS, **over})

    def test_the_harness_and_the_refiner_take_them(self):
        self.assertEqual(lap_sim.parse_driver_override('curvature_smooth=13'),
                         ('curvature_smooth', 13))
        self.assertEqual(lap_sim.parse_driver_override('gg_exponent=3'), ('gg_exponent', 3.0))
        self.assertIn('gg_exponent', raceline.PROFILE_KEYS)
        self.assertIn('curvature_smooth', raceline.PROFILE_KEYS)


if __name__ == '__main__':
    unittest.main()
