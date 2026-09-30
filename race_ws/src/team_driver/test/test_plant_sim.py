"""Tests pinning `plant_sim.GymPlant` to the plant the judge actually runs.

The point of this file is one number: the steering servo saturates at
3.2 rad/s, so a full-lock reversal takes ~290 ms -- longer than the MPC's
entire 200 ms horizon. `lap_sim.py`'s `VehicleModel` applies steering
instantly and therefore reports every one of those manoeuvres as free. That
blindness is why the 2026-09-28 weight sweep passed offline and still
chattered on the judge.

So these tests do two jobs:
  * pin the servo against hand-computed arithmetic, so a refactor cannot
    quietly restore the instant-steering behaviour, and
  * pin the *structure* the fidelity depends on -- controls evaluated once per
    step and held across RK4 sub-stages, and the numba stub staying inert.
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

import plant_sim  # noqa: E402
from plant_sim import BETA, DELTA, GymPlant, PSI, PSI_DOT, V, X, Y  # noqa: E402

DT = 0.01
# 10 * sv_max / (s_max - s_min) = 10 * 3.2 / 0.8378, from action.py:47-48.
EXPECTED_KP = 38.195273334924806
# Steering error above which the P controller is rate-limited: sv_max / kp.
SATURATION_BAND = 3.2 / EXPECTED_KP
# The MPC plans 8 steps at 0.025 s. A manoeuvre slower than this is one the
# controller cannot see the end of.
MPC_HORIZON_S = 0.2


class TestServo(unittest.TestCase):
    """The saturated P steering servo -- root cause RC2."""

    def setUp(self):
        self.plant = GymPlant()

    def test_default_gain_matches_the_judge(self):
        self.assertAlmostEqual(self.plant.steer_kp, EXPECTED_KP, places=9)
        self.assertAlmostEqual(SATURATION_BAND, 0.08378, places=5)

    def test_saturated_step_is_exactly_sv_max_dt(self):
        """Full lock from centre: error 0.4189 rad >> the 0.0838 band."""
        self.plant.reset((0.0, 0.0, 0.0))
        self.plant.latch(0.4189, 0.0)
        self.plant.step()

        self.assertAlmostEqual(self.plant.sv_last, 3.2, places=9)
        self.assertTrue(self.plant.is_rate_saturated)
        # DELTA_DOT is the control itself and the control is held constant, so
        # RK4 integrates it exactly: 3.2 * 0.01.
        self.assertAlmostEqual(float(self.plant.state[DELTA]), 0.032, places=6)

    def test_small_error_is_linear_not_saturated(self):
        """Below the band the servo is a plain P controller.

        This is what makes `sv_last` a meaningful duty signal: if the clip were
        always active, saturation duty would be 100% for any input and could
        not distinguish a chattering config from a smooth one.
        """
        self.plant.reset((0.0, 0.0, 0.0))
        self.plant.latch(0.01, 0.0)
        self.plant.step()

        self.assertAlmostEqual(self.plant.sv_last, EXPECTED_KP * 0.01, places=6)
        self.assertLess(abs(self.plant.sv_last), 3.2)
        self.assertFalse(self.plant.is_rate_saturated)

    def test_full_lock_reversal_outlasts_the_mpc_horizon(self):
        """+0.4189 -> -0.4189 takes longer than the controller can plan for."""
        self.plant.reset_state([0.0, 0.0, 0.4189, 1.5, 0.0, 0.0, 0.0])
        self.plant.latch(-0.4189, 1.5)

        saturated_steps = 0
        largest_move = 0.0
        previous = float(self.plant.state[DELTA])
        elapsed = None
        for step in range(1, 200):
            self.plant.step()
            current = float(self.plant.state[DELTA])
            largest_move = max(largest_move, abs(current - previous))
            previous = current
            if self.plant.is_rate_saturated:
                saturated_steps += 1
            if elapsed is None and abs(current + 0.4189) < 0.01:
                elapsed = step * DT

        self.assertIsNotNone(elapsed, 'the servo never completed the reversal')
        self.assertGreater(
            elapsed, MPC_HORIZON_S,
            'a full-lock reversal now fits inside the MPC horizon -- the '
            'steering servo is no longer rate-limited, so this harness has '
            'stopped modelling the judged plant')
        # 0.8378 rad of travel at 3.2 rad/s, less the unsaturated tail.
        self.assertGreater(saturated_steps, 20)
        # The hard guarantee: the wheel cannot move faster than its rate limit.
        self.assertLessEqual(largest_move, 3.2 * DT + 1e-6)


class TestDynamics(unittest.TestCase):
    """The single-track model itself."""

    def test_launch_from_rest_crosses_the_kinematic_branch(self):
        """`vehicle_dynamics_st` switches branches at V < 0.5.

        The ST branch divides by V, so the low-speed branch is not an
        optimisation -- it is the only thing keeping the launch finite. The car
        starts from rest on every run, so this path is always exercised.
        """
        plant = GymPlant()
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.2, 1.5)

        crossed_slow = False
        for _ in range(400):
            plant.step()
            if float(plant.state[V]) < 0.5:
                crossed_slow = True
            self.assertTrue(np.all(np.isfinite(plant.state)),
                            f'non-finite state during launch: {plant.state}')

        self.assertTrue(crossed_slow, 'never sampled the V < 0.5 branch')
        self.assertAlmostEqual(float(plant.state[V]), 1.5, places=2)

    def test_constant_steer_circle_matches_kinematic_radius(self):
        """A settled 0.2 rad turn at 1.5 m/s traces L/tan(delta), within 2%."""
        plant = GymPlant()
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.2, 1.5)
        for _ in range(400):          # reach steady state
            plant.step()

        xs, ys = [], []
        for _ in range(1200):
            plant.step()
            xs.append(float(plant.state[X]))
            ys.append(float(plant.state[Y]))
        xs, ys = np.array(xs[-900:]), np.array(ys[-900:])

        radius = float(np.hypot(xs - xs.mean(), ys - ys.mean()).mean())
        wheelbase = float(plant.params.lf + plant.params.lr)
        expected = wheelbase / math.tan(0.2)
        self.assertLess(abs(radius - expected) / expected, 0.02,
                        f'radius {radius:.4f} m vs kinematic {expected:.4f} m')

    def test_sideslip_is_nonzero_in_a_turn(self):
        """The MPC assumes beta = 0. The plant disagrees, and says so on odom."""
        plant = GymPlant()
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.2, 1.5)
        for _ in range(600):
            plant.step()

        self.assertGreater(abs(float(plant.state[BETA])), 0.01)
        lin_x, lin_y, ang_z = plant.odom_twist
        speed = float(plant.state[V])
        beta = float(plant.state[BETA])
        self.assertAlmostEqual(lin_x, speed * math.cos(beta), places=6)
        self.assertAlmostEqual(lin_y, speed * math.sin(beta), places=6)
        self.assertAlmostEqual(ang_z, float(plant.state[PSI_DOT]), places=6)


class TestStepStructure(unittest.TestCase):
    """The details that decide whether this harness is honest."""

    def test_controls_evaluated_once_per_step(self):
        """`simulator.py:323-336` holds the control across every RK4 stage.

        Re-evaluating the servo per sub-stage is a different, better-damped
        plant -- it would smooth away the rate saturation this module exists to
        measure. With substeps=2 the naive implementation calls it twice.
        """
        plant = GymPlant(timestep=0.01, integrator_dt=0.005)
        self.assertEqual(plant.substeps, 2)
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.4189, 1.0)

        real_pid_steer = plant_sim.pid_steer
        calls = []

        def counting_pid_steer(*args, **kwargs):
            calls.append(args)
            return real_pid_steer(*args, **kwargs)

        plant_sim.pid_steer = counting_pid_steer
        try:
            plant.step()
        finally:
            plant_sim.pid_steer = real_pid_steer

        self.assertEqual(len(calls), 1,
                         'the steering servo was re-evaluated inside the '
                         'integrator; the plant is no longer the judged one')

    def test_timestep_must_be_a_multiple_of_the_integrator_step(self):
        with self.assertRaises(ValueError):
            GymPlant(timestep=0.01, integrator_dt=0.003)

    def test_substeps_preserve_the_step(self):
        """Halving the integrator step must not change the plant materially.

        RK4 at 0.01 s is already converged for this vehicle, so the finer step
        is a check on the integration, not a better answer. Position is
        compared loosely because 2 s at 2 m/s on a 1.07 m radius is most of
        two laps of a tight circle, where a sub-millimetre speed difference
        shows up as phase; the state variables are compared tightly, which is
        where a genuine integration error would appear first.
        """
        coarse = GymPlant(timestep=0.01, integrator_dt=0.01)
        fine = GymPlant(timestep=0.01, integrator_dt=0.005)
        for plant in (coarse, fine):
            plant.reset((0.0, 0.0, 0.0))
            plant.latch(0.3, 2.0)
            for _ in range(200):
                plant.step()

        for index, name in ((V, 'speed'), (DELTA, 'steering'),
                            (PSI_DOT, 'yaw rate'), (BETA, 'sideslip')):
            self.assertAlmostEqual(float(coarse.state[index]),
                                   float(fine.state[index]), places=4,
                                   msg=f'{name} diverged between integrator steps')

        drift = math.hypot(float(coarse.state[X]) - float(fine.state[X]),
                           float(coarse.state[Y]) - float(fine.state[Y]))
        self.assertLess(drift, 5e-3,
                        f'position drifted {drift * 1e3:.2f} mm over 2 s')


class TestCommandInterface(unittest.TestCase):
    """`latch` and the frame conversions, as the bridge does them."""

    def test_steering_is_clipped_to_the_lock(self):
        """gym_bridge.py:595 clips steering. It does NOT clip speed."""
        plant = GymPlant()
        plant.latch(10.0, 99.0)
        self.assertAlmostEqual(plant.steer_cmd, float(plant.params.s_max), places=9)
        self.assertAlmostEqual(plant.speed_cmd, 99.0, places=9)

        plant.latch(-10.0, -99.0)
        self.assertAlmostEqual(plant.steer_cmd, float(plant.params.s_min), places=9)

    def test_command_is_held_across_steps(self):
        """The driver publishes at ~40 Hz; the plant steps at 100 Hz."""
        plant = GymPlant()
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.4189, 0.0)
        for _ in range(3):
            plant.step()
        # Three saturated steps, no new command needed.
        self.assertAlmostEqual(float(plant.state[DELTA]), 3 * 0.032, places=5)

    def test_base_link_pose_round_trips_through_the_cog(self):
        """`reset` takes base_link; `base_link_pose` gives it back."""
        plant = GymPlant()
        for pose in ((0.0, 0.0, 0.0), (-1.68, -0.01, 0.0), (3.5, -2.25, 1.1)):
            plant.reset(pose)
            back = plant.base_link_pose
            for got, want, axis in zip(back, pose, 'xy0'):
                self.assertAlmostEqual(got, want, places=5, msg=f'{axis} of {pose}')

    def test_reset_offsets_the_cog_forward(self):
        """base_link is the rear axle, so the CoG sits lr ahead of it."""
        plant = GymPlant()
        plant.reset((0.0, 0.0, 0.0))
        self.assertAlmostEqual(float(plant.state[X]), float(plant.params.lr), places=6)
        self.assertAlmostEqual(float(plant.state[Y]), 0.0, places=6)
        self.assertAlmostEqual(float(plant.state[PSI]), 0.0, places=6)


class TestKinematicPlant(unittest.TestCase):
    """The kinematic ablation: the judge's servo, no tyre slip dynamics."""

    def _steady_turn(self, delta=0.2, v=2.0, steps=300):
        plant = plant_sim.KinematicPlant()
        plant.reset_state([0.0, 0.0, delta, v, 0.0, 0.0, 0.0])
        plant.latch(delta, v)
        for _ in range(steps):
            plant.step()
        return plant

    def test_dynamics_come_from_the_external_tree(self):
        source = sys.modules[plant_sim.vehicle_dynamics_ks_cog.__module__].__file__
        self.assertIn(os.path.join('external', 'f1tenth_gym'), source)

    def test_yaw_rate_and_sideslip_are_the_kinematic_ones(self):
        plant = self._steady_turn()
        p = plant.params
        wheelbase = p.lf + p.lr
        beta = math.atan(math.tan(0.2) * p.lr / wheelbase)
        expected = 2.0 * math.cos(beta) * math.tan(0.2) / wheelbase
        self.assertAlmostEqual(float(plant.state[PSI_DOT]), expected, places=4)
        self.assertAlmostEqual(float(plant.state[BETA]), beta, places=5)

    def test_servo_is_still_rate_limited(self):
        plant = plant_sim.KinematicPlant()
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.4, 1.0)
        plant.step()
        self.assertTrue(plant.is_rate_saturated)
        self.assertAlmostEqual(float(plant.state[DELTA]),
                               float(plant.params.sv_max) * plant.timestep, places=5)

    def test_state_keeps_the_judged_seven_state_float32_layout(self):
        plant = self._steady_turn(steps=5)
        self.assertEqual(plant.state.shape, (7,))
        self.assertEqual(plant.state.dtype, np.float32)

    def test_integrates_in_float32_like_the_judge(self):
        seen = []
        real = plant_sim.rk4_integration

        def spy(f, x, u, dt, params):
            seen.append(np.asarray(x).dtype)
            return real(f, x, u, dt, params)

        plant = plant_sim.KinematicPlant()
        plant.reset((0.0, 0.0, 0.0))
        plant.latch(0.1, 1.0)
        orig, plant_sim.rk4_integration = plant_sim.rk4_integration, spy
        try:
            plant.step()
        finally:
            plant_sim.rk4_integration = orig
        self.assertEqual(seen, [np.float32])


class TestOdomHelpers(unittest.TestCase):
    """The frames a stored snapshot is replayed into."""

    def test_base_link_sits_lr_behind_the_cog_along_the_heading(self):
        state = np.array([1.0, 2.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0])
        x, y, yaw = plant_sim.base_link_pose_of(state, 0.2)
        self.assertAlmostEqual(x, 1.0, places=9)
        self.assertAlmostEqual(y, 1.8, places=9)
        self.assertAlmostEqual(yaw, math.pi / 2, places=9)

    def test_twist_splits_speed_by_sideslip_and_carries_yaw_rate(self):
        state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.7, 0.1])
        vx, vy, wz = plant_sim.odom_twist_of(state)
        self.assertAlmostEqual(vx, 2.0 * math.cos(0.1), places=6)
        self.assertAlmostEqual(vy, 2.0 * math.sin(0.1), places=6)
        self.assertAlmostEqual(wz, 0.7, places=6)


class TestGymImport(unittest.TestCase):
    """The plant must be the judge's code, not a copy of it."""

    def test_dynamics_come_from_the_external_tree(self):
        source = sys.modules[plant_sim.vehicle_dynamics_st.__module__].__file__
        self.assertIn(os.path.join('external', 'f1tenth_gym'), source)

    def test_vehicle_params_are_the_f1tenth_preset(self):
        params = plant_sim.F1TENTH_VEHICLE_PARAMETERS
        self.assertAlmostEqual(params.sv_max, 3.2, places=9)
        self.assertAlmostEqual(params.s_max, 0.4189, places=9)
        self.assertAlmostEqual(params.lf, 0.15875, places=9)
        self.assertAlmostEqual(params.lr, 0.17145, places=9)

    def test_numba_stub_is_a_pure_passthrough(self):
        """If numba is stubbed, `njit` must not alter the function at all."""
        njit = sys.modules['numba'].njit

        def sample(a, b):
            return a + b

        self.assertIs(njit(sample), sample)
        self.assertIs(njit(cache=True)(sample), sample)

    def test_stub_yields_to_a_real_numba(self):
        """Installing numba later must silently take over."""
        self.assertFalse(plant_sim._install_numba_stub(),
                         'a second stub was installed over the first')


if __name__ == '__main__':
    unittest.main()
