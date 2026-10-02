"""Step 4: the plant's proportional speed loop, inverted."""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import plant_sim  # noqa: E402
from vehicle_model import SpeedLoop, VehicleParameters  # noqa: E402

GYM_UTILS = sys.modules['f1tenth_gym.envs.dynamic_models.utils']
P = plant_sim.F1TENTH_VEHICLE_PARAMETERS
LOOP = SpeedLoop.of(VehicleParameters())
HOLD = 0.025                      # one control period
PHYSICS_DT = 0.01


def _plant_at(v0):
    plant = plant_sim.GymPlant()
    plant.reset_state([0.0, 0.0, 0.0, v0, 0.0, 0.0, 0.0])
    return plant


def _gym_accel(v_cmd, v):
    """What the judged plant applies: pid_accl, then accl_constraints."""
    raw = plant_sim.pid_accl(v_cmd, v, P.a_max, P.v_max, P.v_min)
    return GYM_UTILS.accl_constraints(v, raw, P.v_switch, P.a_max, P.v_min, P.v_max)


class GainTest(unittest.TestCase):

    def test_the_gains_are_derived_from_the_vehicle_parameters(self):
        self.assertAlmostEqual(LOOP.k_accel, 4.755, places=12)
        self.assertAlmostEqual(LOOP.k_brake, 19.02, places=12)

    def test_the_gains_are_the_ones_pid_accl_applies_when_moving_forward(self):
        self.assertAlmostEqual(plant_sim.pid_accl(3.1, 3.0, P.a_max, P.v_max, P.v_min),
                               LOOP.k_accel * 0.1, places=9)
        self.assertAlmostEqual(plant_sim.pid_accl(2.9, 3.0, P.a_max, P.v_max, P.v_min),
                               -LOOP.k_brake * 0.1, places=9)

    def test_the_loop_applies_what_the_plant_applies_everywhere(self):
        # Every quadrant of pid_accl (including the weaker gains at rest and in
        # reverse), the 9.51 m/s^2 clamp and the fall-off above v_switch.
        for v in (-1.0, 0.0, 0.5, 3.0, 7.0, 9.0, 15.0):
            for v_cmd in (-2.0, 0.0, v - 0.3, v + 0.05, v + 3.0, 12.0):
                with self.subTest(v=v, v_cmd=v_cmd):
                    self.assertAlmostEqual(LOOP.accel(v_cmd, v), _gym_accel(v_cmd, v),
                                           places=9)


class ValidationTest(unittest.TestCase):

    def test_a_loop_the_plant_could_not_have_is_refused(self):
        for bad in (dict(step=0.0), dict(a_max=0.0), dict(v_max=-1.0), dict(v_min=1.0)):
            with self.subTest(**bad), self.assertRaises(ValueError):
                SpeedLoop(**bad)


class InversionTest(unittest.TestCase):

    def test_no_acceleration_asks_for_the_speed_the_car_has(self):
        self.assertEqual(LOOP.command(4.0, 0.0, HOLD), 4.0)

    def test_the_setpoint_leads_the_speed_when_accelerating_and_trails_it_when_braking(self):
        self.assertGreater(LOOP.command(4.0, 3.0, HOLD), 4.0 + 3.0 * HOLD)
        self.assertLess(LOOP.command(4.0, -3.0, HOLD), 4.0 - 3.0 * HOLD)

    def test_for_a_short_hold_the_inversion_is_a_over_the_gain(self):
        # The plan's v + a/k is the limit of the exact inversion as the hold
        # shrinks to nothing.
        self.assertAlmostEqual(LOOP.command(4.0, 2.0, 1e-7), 4.0 + 2.0 / LOOP.k_accel, places=5)
        self.assertAlmostEqual(LOOP.command(4.0, -2.0, 1e-7), 4.0 - 2.0 / LOOP.k_brake, places=5)

    def test_the_plant_achieves_the_commanded_acceleration_over_a_hold(self):
        # A 25 ms command on the 10 ms grid is held for 2 then 3 steps.
        for v0 in (2.0, 4.0, 6.0, 8.0):
            for a_des in (3.0, 6.0, -3.0, -6.0, -9.0):
                for steps in (2, 3):
                    with self.subTest(v0=v0, a_des=a_des, steps=steps):
                        plant = _plant_at(v0)
                        plant.latch(0.0, LOOP.command(v0, a_des, HOLD))
                        for _ in range(steps):
                            plant.step()
                        achieved = (float(plant.state[plant_sim.V]) - v0) / (steps * PHYSICS_DT)
                        self.assertAlmostEqual(achieved / a_des, 1.0, delta=0.10)

    def test_a_sustained_demand_is_achieved_in_closed_loop(self):
        for v0, a_des in ((2.0, 4.0), (7.0, -6.0)):
            with self.subTest(v0=v0, a_des=a_des):
                plant = _plant_at(v0)
                t = 0.0
                for k in range(20):                   # 0.5 s of 25 ms commands
                    plant.latch(0.0, LOOP.command(float(plant.state[plant_sim.V]),
                                                  a_des, HOLD))
                    steps = 3 if k % 2 == 0 else 2
                    for _ in range(steps):
                        plant.step()
                    t += steps * PHYSICS_DT
                achieved = (float(plant.state[plant_sim.V]) - v0) / t
                self.assertAlmostEqual(achieved / a_des, 1.0, delta=0.10)

    def test_a_demand_past_the_plant_limit_leads_no_further_than_the_limit(self):
        # Above v_switch the plant cannot give 9.51 m/s^2; a bigger lead would
        # only be a bigger step for the next command to take back.
        v = 10.0
        ceiling = LOOP.accel_ceiling(v)
        self.assertLess(ceiling, P.a_max)
        self.assertEqual(LOOP.command(v, 50.0, HOLD), LOOP.command(v, ceiling, HOLD))
        self.assertEqual(LOOP.command(v, -50.0, HOLD), LOOP.command(v, -P.a_max, HOLD))


if __name__ == '__main__':
    unittest.main()
