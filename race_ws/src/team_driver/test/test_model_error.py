"""Phase 2 Step 3: which model predicts the judged plant better, measured.

`model_error.py` drives the judged plant (GymPlant) through a steering sweep at
a held speed and asks each MPC model to predict it from odometry's state: over
one control period, and over a horizon of real commands. The plan keeps
whichever model predicts better; this pins the measurement and the verdict.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import model_error  # noqa: E402

SECONDS = 2.0


class TraceTest(unittest.TestCase):

    def test_the_plant_is_held_near_the_requested_speed(self):
        trace = model_error.plant_trace(5.0, SECONDS)
        speeds = [tick.state[model_error.plant_sim.V] for tick in trace]
        self.assertAlmostEqual(min(speeds), 5.0, delta=0.3)
        self.assertAlmostEqual(max(speeds), 5.0, delta=0.3)

    def test_ticks_follow_the_control_period(self):
        trace = model_error.plant_trace(3.0, SECONDS)
        dts = [b.t - a.t for a, b in zip(trace, trace[1:])]
        self.assertAlmostEqual(sum(dts) / len(dts), 0.025, delta=0.001)


class ErrorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = {(r.speed, r.model): r
                    for r in model_error.error_table((3.0, 5.0, 7.0), SECONDS)}

    def test_every_speed_and_model_is_measured(self):
        self.assertEqual(len(self.rows), 6)

    def test_single_track_predicts_one_step_better_at_racing_speed(self):
        for v in (5.0, 7.0):
            with self.subTest(v=v):
                self.assertLess(self.rows[(v, 'ST')].step_pos_mean,
                                self.rows[(v, 'KS')].step_pos_mean)

    def test_single_track_predicts_the_horizon_better_at_racing_speed(self):
        for v in (5.0, 7.0):
            with self.subTest(v=v):
                self.assertLess(self.rows[(v, 'ST')].horizon_pos_mean,
                                self.rows[(v, 'KS')].horizon_pos_mean)

    def test_single_track_one_step_error_is_sub_millimetre(self):
        for v in (3.0, 5.0, 7.0):
            with self.subTest(v=v):
                self.assertLess(self.rows[(v, 'ST')].step_pos_max, 1e-3)

    def test_the_table_prints(self):
        text = model_error.format_table(list(self.rows.values()))
        self.assertIn('ST', text)
        self.assertIn('KS', text)


if __name__ == '__main__':
    unittest.main()
