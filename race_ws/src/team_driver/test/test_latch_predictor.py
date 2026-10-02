"""Step 3.5: the plant replayed over the dead time, to the next latch."""

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
import plant_sim  # noqa: E402
from latch_predictor import HISTORY_SPAN, LatchPredictor, servo_kp  # noqa: E402
from vehicle_model import VehicleParameters  # noqa: E402

VP = VehicleParameters()
PHYSICS_DT = 0.01


def _plant(v, delta=0.0, psi=0.0, beta=0.0, yaw_rate=0.0):
    plant = plant_sim.GymPlant()
    plant.reset_state([1.0, 2.0, delta, v, psi, yaw_rate, beta])
    return plant


def _measured(plant):
    """The ST state the driver holds: base_link pose, odom speed, slip, wheel, yaw rate."""
    x, y, yaw = plant.base_link_pose
    s = plant.state
    return (x, y, yaw, float(s[plant_sim.V]), float(s[plant_sim.BETA]),
            float(s[plant_sim.DELTA]), float(s[plant_sim.PSI_DOT]))


def _settled_corner(v=5.0, steer=0.1):
    """A plant that has held a steady turn for 2 s, its clock reset to zero."""
    plant = _plant(v)
    plant.latch(steer, v)
    for _ in range(200):
        plant.step()
    plant.time = 0.0
    return plant


class AccuracyTest(unittest.TestCase):

    def _check(self, plant, schedule, until, history=None):
        predictor = LatchPredictor(VP)
        for t, steer, speed in (history if history is not None else schedule):
            predictor.on_publish(steer, speed, t)
        start = _measured(plant)
        prev = plant.base_link_pose[:2]
        driven = 0.0
        pending = [s for s in schedule if s[0] >= 0.0]
        t = 0.0
        while t < until - 1e-9:
            while pending and pending[0][0] <= t + 1e-9:
                _, steer, speed = pending.pop(0)
                plant.latch(steer, speed)
            plant.step()
            t += PHYSICS_DT
            here = plant.base_link_pose[:2]
            driven += math.hypot(here[0] - prev[0], here[1] - prev[1])
            prev = here
        predicted = predictor.predict(start, 0.0, until)
        actual = _measured(plant)
        np.testing.assert_allclose(predicted.state[:2], actual[:2], atol=0.005)    # position
        self.assertAlmostEqual(predicted.state[2], actual[2], delta=0.002)       # yaw
        self.assertAlmostEqual(predicted.state[3], actual[3], delta=0.002)       # speed
        self.assertAlmostEqual(predicted.state[4], actual[4], delta=0.005)       # slip
        self.assertAlmostEqual(predicted.state[5], actual[5], delta=1e-4)        # wheel
        self.assertAlmostEqual(predicted.state[6], actual[6], delta=0.05)        # yaw rate
        self.assertAlmostEqual(predicted.travelled, driven, delta=0.005)
        return predicted

    def test_a_steady_turn_is_replayed(self):
        self._check(_settled_corner(), [(0.0, 0.1, 5.0)], 0.15)

    def test_steering_and_speed_changes_are_replayed(self):
        # Turn in, brake, turn back the other way: the dead time a corner entry
        # spends, at the plant's own latch times.
        schedule = [(0.0, 0.2, 6.0), (0.03, -0.05, 3.5), (0.08, 0.15, 4.5)]
        self._check(_settled_corner(), schedule, 0.15)

    def test_a_command_latched_before_the_pose_is_the_one_acting(self):
        plant = _settled_corner()
        plant.latch(-0.1, 3.0)
        history = [(-0.04, -0.1, 3.0), (0.05, 0.1, 6.0)]
        self._check(plant, [(0.05, 0.1, 6.0)], 0.12, history=history)

    def test_the_steering_is_clipped_to_the_lock_as_the_bridge_does(self):
        predictor = LatchPredictor(VP)
        predictor.on_publish(2.0, 3.0, 0.0)
        state = predictor.predict((0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0), 0.0, 1.0).state
        self.assertLessEqual(state[5], VP.s_max + 1e-12)

    def test_the_servo_gain_is_the_plants(self):
        self.assertAlmostEqual(servo_kp(VP), 10.0 * 3.2 / (2.0 * 0.4189), places=9)


class BookkeepingTest(unittest.TestCase):
    STATE = (0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0)

    def test_with_no_command_known_wheel_and_speed_are_held(self):
        p = LatchPredictor(VP).predict(self.STATE, 0.0, 0.1)
        self.assertEqual(p.speed, 4.0)
        self.assertAlmostEqual(p.travelled, 0.4, places=9)
        self.assertAlmostEqual(p.state[0], 0.4, places=9)

    def test_no_interval_predicts_nothing(self):
        predictor = LatchPredictor(VP)
        predictor.on_publish(0.1, 9.0, 0.0)
        for t0, t1 in ((0.1, 0.1), (0.1, 0.05), (None, 0.1), (0.1, None)):
            with self.subTest(t0=t0, t1=t1):
                self.assertEqual(predictor.predict(self.STATE, t0, t1),
                                 (self.STATE, 0.0))

    def test_a_newer_command_supersedes_ones_due_at_or_after_it(self):
        predictor = LatchPredictor(VP)
        predictor.on_publish(0.3, 9.0, 0.05)
        predictor.on_publish(0.0, 3.0, 0.02)        # published later, latches earlier
        expected = LatchPredictor(VP)
        expected.on_publish(0.0, 3.0, 0.02)
        self.assertEqual(predictor.predict(self.STATE, 0.0, 0.1),
                         expected.predict(self.STATE, 0.0, 0.1))

    def test_a_command_with_no_latch_time_is_not_placed(self):
        predictor = LatchPredictor(VP)
        predictor.on_publish(0.3, 9.0, None)
        self.assertEqual(len(predictor), 0)

    def test_reset_forgets_the_commands(self):
        predictor = LatchPredictor(VP)
        predictor.on_publish(0.0, 9.0, 0.0)
        predictor.reset()
        self.assertEqual(predictor.predict(self.STATE, 0.0, 0.1).speed, 4.0)

    def test_history_stays_bounded_and_keeps_the_acting_command(self):
        predictor = LatchPredictor(VP)
        for k in range(10_000):
            predictor.on_publish(0.0, 5.0, k * 0.025)
        self.assertLess(len(predictor), 100)
        t_last = 9_999 * 0.025
        v = predictor.predict((0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0), t_last - 0.2, t_last).speed
        self.assertGreater(v, 4.0)
        self.assertLessEqual(v, 5.0)

    def test_a_replay_longer_than_the_history_stops_at_its_span(self):
        p = LatchPredictor(VP).predict(self.STATE, 0.0, 50.0)
        self.assertAlmostEqual(p.travelled, 4.0 * HISTORY_SPAN, places=9)


class _RecordingMPC:
    """Just enough of MPCController for DriverCore.plan; records the state."""

    dt = 0.025
    state_dim = 7

    def __init__(self):
        self.states = []

    def speed_profile(self, **_kw):
        return np.full(4, 3.0)

    def set_reference_path(self, *_a):
        pass

    def get_control(self, state):
        self.states.append(np.asarray(state, dtype=float))
        return np.array([0.0, 0.0]), {'solve_time_ms': 0.0}


class CoreWiringTest(unittest.TestCase):
    """The MPC plans from where the car will be when its command latches."""

    def _second_state(self, predict_latch):
        mpc = _RecordingMPC()
        core = driver_core.DriverCore(
            cfg=driver_core.DriverConfig(predict_latch=predict_latch), mpc_controller=mpc)
        core.reference_path = np.zeros((4, 2))
        scan = (np.full(181, 6.0), -np.pi / 2, np.pi / 180)
        core.set_odom(0.0, 0.0, 0.0, 4.0, t_sim=0.0)
        core.set_scan(*scan, t_sim=0.0)
        core.step(t_sim=0.0)
        # A faster command latched at 0.09; the pose for the next plan is from
        # 0.02, so the next command lands at 0.115, ~0.1 s of dead time.
        core.note_published(driver_core.DriveCommand(0.0, 6.0), t_sim=0.09)
        core.set_odom(0.08, 0.0, 0.0, 4.0, t_sim=0.02)
        core.set_scan(*scan, t_sim=0.02)
        core.step(t_sim=0.02)
        return mpc.states[-1], core

    def test_the_mpc_is_given_the_state_at_the_latch(self):
        state, core = self._second_state(True)
        # 0.095 s from the pose: held at 4 m/s to 0.09, then speeding up.
        self.assertGreater(state[0], 0.08 + 4.0 * 0.095 - 1e-6)
        self.assertGreater(state[3], 4.0)
        self.assertAlmostEqual(core._latch_travel(), state[0] - 0.08, places=6)

    def test_with_prediction_off_it_is_given_the_measured_state(self):
        state, core = self._second_state(False)
        self.assertEqual((state[0], state[3]), (0.08, 4.0))
        self.assertEqual(core._latch_travel(), 0.0)


if __name__ == '__main__':
    unittest.main()
