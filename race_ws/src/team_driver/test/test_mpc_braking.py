"""Sub-9 Step 2: the MPC owns the braking, keeps its slip state, and the profile stops overriding it.

Every crash above a_lat_max 22 was a braking-yaw spin: the MPC sat at its
-0.3 a_max brake floor while the profile override braked -5 to -7.8 m/s^2 past
it, unseen by the prediction. The defaults must keep the old car exactly.
"""

import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import driver_core  # noqa: E402
import driver_params  # noqa: E402
import lap_sim  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

DT = 0.025
# The pre-sub-9 car: these three settings reproduce 6c3d650's laps bit for bit.
OLD_CAR = dict(mpc_full_brake=False, profile_brake_cap=9.51, beta_limit=0.5)


def _core(**cfg):
    mpc = MPCController(vehicle_model=VehicleModel(), **lap_sim.JUDGED_MPC)
    return driver_core.DriverCore(cfg=driver_core.DriverConfig(**cfg), mpc_controller=mpc)


class DefaultsTest(unittest.TestCase):
    def test_the_defaults_are_the_sub9_pin(self):
        cfg = driver_core.DriverConfig()
        self.assertTrue(cfg.mpc_full_brake)
        self.assertEqual(cfg.profile_brake_cap, 3.0)
        self.assertEqual(cfg.beta_limit, 1.0)


class MpcBrakeFloorTest(unittest.TestCase):
    def test_the_old_car_brakes_to_its_old_floor(self):
        core = _core(**OLD_CAR)
        p = core.mpc_controller.vehicle_model.params
        self.assertEqual(core.mpc_controller.control_min[1], p.a_max * -0.3)

    def test_full_brake_lets_the_mpc_plan_a_brake_max(self):
        core = _core(mpc_full_brake=True, a_brake_max=6.0)
        self.assertEqual(core.mpc_controller.control_min[1], -6.0)

    def test_full_brake_leaves_the_steering_bound_alone(self):
        before = _core(**OLD_CAR).mpc_controller.control_min[0]
        self.assertEqual(_core(mpc_full_brake=True).mpc_controller.control_min[0], before)

    def test_the_vehicle_models_bounds_are_not_mutated(self):
        model = VehicleModel()
        mpc = MPCController(vehicle_model=model, **lap_sim.JUDGED_MPC)
        driver_core.DriverCore(cfg=driver_core.DriverConfig(), mpc_controller=mpc)
        self.assertEqual(model.get_control_bounds()[0][1], model.params.a_max * -0.3)


class BetaLimitTest(unittest.TestCase):
    def _beta_seen(self, beta, **cfg):
        core = _core(**cfg)
        core.set_odom(0.0, 0.0, 0.0, 5.0, beta=beta, yaw_rate=0.0, t_sim=0.0)
        return core._measured_state()[4]

    def test_in_the_old_car_a_slip_past_half_a_radian_reads_zero(self):
        self.assertEqual(self._beta_seen(0.7, beta_limit=0.5), 0.0)
        self.assertEqual(self._beta_seen(0.4, beta_limit=0.5), 0.4)

    def test_a_wider_limit_keeps_the_slip(self):
        self.assertEqual(self._beta_seen(0.7, beta_limit=1.0), 0.7)
        self.assertEqual(self._beta_seen(-1.2, beta_limit=1.0), 0.0)


class ProfileBrakeCapTest(unittest.TestCase):
    """The profile term may brake at most `profile_brake_cap`; the emergency cap is untouched."""

    V = 8.0

    def _condition(self, accel, profile_speed, speed_cap=math.inf, **cfg):
        core = _core(**cfg)
        core.speed = self.V
        return core, core._condition(0.0, accel, DT, profile_speed=profile_speed,
                                     speed_cap=speed_cap)

    def test_in_the_old_car_the_profile_brakes_to_the_plant_limit(self):
        core, cmd = self._condition(0.0, self.V - 1.0, **OLD_CAR)
        self.assertAlmostEqual(cmd.speed, core._setpoint(-9.51))

    def test_a_cap_holds_the_profile_braking(self):
        core, cmd = self._condition(0.0, self.V - 1.0, profile_brake_cap=3.0)
        self.assertAlmostEqual(cmd.speed, core._setpoint(-3.0))

    def test_a_zero_cap_only_stops_acceleration_above_the_profile(self):
        core, cmd = self._condition(2.0, self.V - 1.0, profile_brake_cap=0.0)
        self.assertAlmostEqual(cmd.speed, core._setpoint(0.0))

    def test_the_mpcs_own_braking_passes_under_a_cap(self):
        core, cmd = self._condition(-5.0, self.V - 1.0, profile_brake_cap=0.0)
        self.assertAlmostEqual(cmd.speed, core._setpoint(-5.0))

    def test_the_emergency_cap_still_brakes_past_it(self):
        core, cmd = self._condition(0.0, self.V - 1.0, speed_cap=self.V - 0.1,
                                    profile_brake_cap=0.0)
        self.assertAlmostEqual(cmd.speed, core._setpoint(-0.1 / DT))


class ValidationTest(unittest.TestCase):
    def _values(self, **over):
        values = dict(driver_params.DEFAULTS)
        values.update(over)
        return values

    def test_the_new_defaults_pass(self):
        driver_params.validate(self._values())

    def test_a_negative_profile_brake_cap_is_refused(self):
        with self.assertRaises(ValueError):
            driver_params.validate(self._values(profile_brake_cap=-1.0))

    def test_a_beta_limit_of_zero_is_refused(self):
        with self.assertRaises(ValueError):
            driver_params.validate(self._values(beta_limit=0.0))

    def test_the_harness_takes_the_new_fields(self):
        self.assertEqual(lap_sim.parse_driver_override('mpc_full_brake=true'),
                         ('mpc_full_brake', True))
        self.assertEqual(lap_sim.parse_driver_override('profile_brake_cap=0'),
                         ('profile_brake_cap', 0.0))


if __name__ == '__main__':
    unittest.main()
