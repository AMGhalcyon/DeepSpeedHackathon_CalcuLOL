"""Phase 2 Step 3: a longer horizon at the same cost, a slip-aware model, feed-forward.

At 1.5 m/s the MPC's 8 x 25 ms horizon looks 0.3 m ahead; at 6 m/s it is 1.2 m,
less than a corner entry. `dt_schedule` stretches the later steps so the same 8
steps (and the same 6 free variables) see ~0.5 s ahead. The first step stays
the control period, because the servo latch and the warm-start shift both
assume a command is held for exactly one step.
"""

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import mpc_controller  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

DT = 0.025
BLOCK_MAP = [0, 1, 1, 2, 2, 2, 2, 2]
HORIZON = len(BLOCK_MAP)
LONG = [0.025, 0.025, 0.05, 0.05, 0.075, 0.075, 0.075, 0.075]
WHEELBASE = VehicleModel().params.lf + VehicleModel().params.lr


def _circle(radius=4.0, n=400, ccw=True):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    sign = 1.0 if ccw else -1.0
    return np.column_stack([radius * np.cos(sign * t), radius * np.sin(sign * t)])


def _controller(model='KS', path=None, **overrides):
    kwargs = dict(vehicle_model=VehicleModel(model_type=model),
                  prediction_horizon=HORIZON, block_map=BLOCK_MAP, dt=DT,
                  r_delta_steer=3.0, solve_budget_ms=1e6)
    kwargs.update(overrides)
    mpc = MPCController(**kwargs)
    path = _circle() if path is None else path
    mpc.set_reference_path(path, np.full(len(path), 3.0))
    return mpc


def _result(sequence):
    return mock.Mock(success=True, status=0, message='stub',
                     x=np.asarray(sequence, dtype=float))


# The commands the controller produced before Phase 2 Step 3 (commit 0458148's
# mpc_controller.py and vehicle_model.py, run on _closed_loop's scenario).
# Pinned so "an all-0.025 schedule is today's controller" compares against
# today's code, not against the new code's own other path.
HEAD_COMMANDS = [
    (0.05642223963027556, 1.8254701769112371),
    (0.08773783822720686, 2.082106854401183),
    (0.10669025805416503, 1.9468862776837845),
    (0.11551230817906372, 1.7364691520050592),
    (0.11701662599459638, 1.5344609743084339),
    (0.11225984121238365, 1.3447256273614363),
    (0.10504504207507628, 1.177348529318869),
    (0.09883125117346026, 1.0307344772820506),
    (0.09369581485783553, 0.904018152447295),
    (0.08975259143891541, 0.7939755040129179),
    (0.08848482346142432, 0.6963852963577337),
    (0.08694858239894358, 0.6120476018807766),
    (0.08485010896995364, 0.5345453729015501),
    (0.0834100973796701, 0.46968450432475967),
    (0.08240233989467664, 0.41232092455935665),
    (0.08347610817064675, 0.36215524626306855),
    (0.0836353414756502, 0.32704592008349204),
    (0.08313686951907034, 0.2893989199641081),
    (0.08320425312216427, 0.25366810898427633),
    (0.08289455474088124, 0.25373367908354993),
    (0.08241341017076774, 0.19688030550349417),
    (0.08221251498324324, 0.18898983779848733),
    (0.08183244307144591, 0.1450711780487397),
    (0.08280082855523603, 0.13855900819797745),
    (0.08334407571665811, 0.12311274170741161),
]
# [rad, m/s^2] The fused objective is the same optimum, not the same bits.
FUSED_COMMAND_TOLERANCE = 1e-4


def _closed_loop(mpc, steps=25):
    """Drive a KS model with the controller's own commands; return them."""
    model = VehicleModel()
    state = np.array([4.0, 0.05, math.pi / 2, 2.5, 0.0, 0.0])
    controls = []
    for _ in range(steps):
        u, _ = mpc.get_control(state)
        controls.append(u.copy())
        state = model.actuated_step(state, u, DT)
    return np.array(controls)


class ScheduleValidationTest(unittest.TestCase):

    def test_no_schedule_is_every_step_the_control_period(self):
        np.testing.assert_array_equal(_controller().dt_schedule, [DT] * HORIZON)

    def test_the_schedule_is_kept(self):
        np.testing.assert_array_equal(_controller(dt_schedule=LONG).dt_schedule, LONG)

    def test_lookahead_is_the_schedule_total(self):
        self.assertAlmostEqual(_controller().lookahead, 0.2)
        self.assertAlmostEqual(_controller(dt_schedule=LONG).lookahead, 0.45)

    def test_a_schedule_of_the_wrong_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'dt_schedule'):
            _controller(dt_schedule=LONG[:-1])

    def test_a_first_step_other_than_the_control_period_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'first'):
            _controller(dt_schedule=[0.05] + LONG[1:])

    def test_a_non_positive_step_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            _controller(dt_schedule=LONG[:3] + [0.0] + LONG[4:])

    def test_a_non_finite_step_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            _controller(dt_schedule=LONG[:3] + [math.nan] + LONG[4:])


class UniformScheduleReproducesTodayTest(unittest.TestCase):
    """An all-0.025 schedule is the controller as it was, to the bit."""

    def _unfused(self, **overrides):
        mpc = _controller(**overrides)
        mpc.fused_objective = False
        return mpc

    def test_closed_loop_commands_are_the_pre_step3_controllers_to_the_bit(self):
        for schedule in (None, [DT] * HORIZON):
            with self.subTest(schedule=schedule):
                np.testing.assert_array_equal(
                    _closed_loop(self._unfused(dt_schedule=schedule)), HEAD_COMMANDS)

    def test_the_fused_objective_lands_on_the_same_commands(self):
        np.testing.assert_allclose(_closed_loop(_controller(dt_schedule=[DT] * HORIZON)),
                                   HEAD_COMMANDS, rtol=0, atol=FUSED_COMMAND_TOLERANCE)

    def test_closed_loop_commands_are_identical(self):
        legacy = _closed_loop(_controller())
        scheduled = _closed_loop(_controller(dt_schedule=[DT] * HORIZON))
        np.testing.assert_array_equal(scheduled, legacy)

    def test_the_reference_window_is_identical(self):
        a = _controller()._reference_window(3.0, HORIZON + 1, car_speed=2.0).copy()
        b = _controller(dt_schedule=[DT] * HORIZON)._reference_window(
            3.0, HORIZON + 1, car_speed=2.0).copy()
        np.testing.assert_array_equal(a, b)

    def test_the_warm_start_is_a_one_step_shift(self):
        mpc = _controller(dt_schedule=[DT] * HORIZON)
        np.testing.assert_array_equal(mpc._shift_index, [1, 2, 3, 4, 5, 6, 7, 7])


class LongScheduleTest(unittest.TestCase):

    def test_each_prediction_step_uses_its_own_length(self):
        mpc = _controller(dt_schedule=LONG)
        seen = []

        def record(state, control, dt):
            seen.append(float(dt))
            return state

        with mock.patch.object(mpc.vehicle_model, 'actuated_step', side_effect=record):
            mpc._predict_trajectory(np.array([0, 0, 0, 1.0, 0, 0]), np.zeros(6))
        self.assertEqual(seen, LONG)

    def test_the_reference_is_where_the_car_will_be_at_each_cumulative_time(self):
        mpc = _controller(dt_schedule=LONG)
        v = 4.0
        window = mpc._reference_window(1.0, HORIZON + 1, car_speed=v)
        times = np.concatenate([[0.0], np.cumsum(LONG)])
        # On a circle the chord between window points fixes the arc between them.
        radius = 4.0
        for k in range(1, HORIZON + 1):
            chord = np.hypot(*(window[k, :2] - window[0, :2]))
            arc = 2.0 * radius * math.asin(min(chord / (2.0 * radius), 1.0))
            self.assertAlmostEqual(arc, v * times[k], delta=0.01)

    def test_the_warm_start_advances_by_one_control_period_of_time(self):
        # Steps start at 0, .025, .05, .1, .15, .225, .3, .375 s; a plan
        # advanced by 25 ms reads each step from the one covering t + 25 ms.
        mpc = _controller(dt_schedule=LONG)
        np.testing.assert_array_equal(mpc._shift_index, [1, 2, 2, 3, 4, 5, 6, 7])

    def test_still_six_free_variables(self):
        mpc = _controller(dt_schedule=LONG)
        with mock.patch.object(mpc_controller, 'minimize',
                               return_value=_result(np.zeros(6))) as spy:
            mpc.get_control(np.array([4.0, 0.0, math.pi / 2, 3.0]))
        self.assertEqual(len(spy.call_args.args[1]), 6)


class SingleTrackControllerTest(unittest.TestCase):

    def test_the_state_grows_by_the_yaw_rate(self):
        self.assertEqual(_controller(model='ST').state_dim, 7)
        self.assertEqual(_controller().state_dim, 6)

    def test_a_full_odometry_state_is_accepted(self):
        mpc = _controller(model='ST', dt_schedule=LONG)
        u, _ = mpc.get_control(np.array([4.0, 0.0, math.pi / 2, 3.0, 0.02, 0.1, 0.7]))
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertIsNone(mpc.last_fallback)

    def test_a_state_without_slip_or_yaw_rate_is_padded(self):
        mpc = _controller(model='ST')
        u, _ = mpc.get_control(np.array([4.0, 0.0, math.pi / 2, 3.0]))
        self.assertTrue(np.all(np.isfinite(u)))

    def test_a_state_of_the_wrong_length_is_refused(self):
        with self.assertRaises(ValueError):
            _controller(model='ST').get_control(np.zeros(8))

    def test_the_heading_cost_compares_the_direction_of_travel(self):
        # A sliding car whose rear axle moves exactly along the path pays no
        # heading cost, though its body points inward of the tangent.
        mpc = _controller(model='ST', path=np.array([[0.0, 0.0], [50.0, 0.0]]))
        v, beta, r = 5.0, 0.05, 0.0
        drift = math.atan2(v * math.sin(beta) - mpc.vehicle_model.params.lr * r,
                           v * math.cos(beta))
        states = np.zeros((HORIZON + 1, 7))
        states[:, 2] = -drift
        states[:, 3] = v
        states[:, 4] = beta
        ref = np.zeros((HORIZON + 1, 7))
        ref[:, 3] = v
        with mock.patch.object(mpc, '_predict_trajectory', return_value=states):
            cost = mpc._cost_function(np.zeros(6), states[0], ref, ref[:, 3])
        self.assertAlmostEqual(cost, 0.0, places=12)

    def test_the_kinematic_heading_cost_is_unchanged(self):
        mpc = _controller(path=np.array([[0.0, 0.0], [50.0, 0.0]]))
        states = np.zeros((HORIZON + 1, 6))
        states[:, 2] = 0.1
        ref = np.zeros((HORIZON + 1, 6))
        with mock.patch.object(mpc, '_predict_trajectory', return_value=states):
            cost = mpc._cost_function(np.zeros(6), states[0], ref, ref[:, 3])
        self.assertAlmostEqual(cost, mpc.q_yaw_wrap * (HORIZON + 1) * 0.01, places=12)


class FeedForwardTest(unittest.TestCase):

    def _first_guess(self, model, ccw=True, feedforward=True, v=3.0):
        mpc = _controller(model=model, path=_circle(ccw=ccw), feedforward=feedforward,
                          dt_schedule=LONG)
        heading = math.pi / 2 if ccw else -math.pi / 2
        with mock.patch.object(mpc_controller, 'minimize',
                               return_value=_result(np.zeros(6))) as spy:
            mpc.get_control(np.array([4.0, 0.0, heading, v]))
        return spy.call_args.args[1], mpc

    def test_off_by_default(self):
        self.assertFalse(_controller().feedforward)

    def test_the_first_move_is_seeded_with_the_corner_geometry(self):
        guess, _ = self._first_guess('KS')
        self.assertAlmostEqual(guess[0], math.atan(WHEELBASE / 4.0), delta=0.01)

    def test_a_right_hander_seeds_a_right_turn(self):
        guess, _ = self._first_guess('KS', ccw=False)
        self.assertAlmostEqual(guess[0], -math.atan(WHEELBASE / 4.0), delta=0.01)

    def test_the_single_track_seed_adds_the_understeer(self):
        guess, mpc = self._first_guess('ST', v=5.0)
        self.assertAlmostEqual(guess[0], mpc.vehicle_model.steady_steer(0.25, 5.0),
                               delta=0.01)

    def test_without_feedforward_the_warm_start_is_the_shifted_plan(self):
        guess, _ = self._first_guess('KS', feedforward=False)
        self.assertEqual(guess[0], 0.0)

    def test_only_the_steering_of_the_first_move_is_seeded(self):
        guess, _ = self._first_guess('KS')
        np.testing.assert_array_equal(guess[1:], np.zeros(5))

    def test_the_seed_respects_the_steering_bounds(self):
        tight = _circle(radius=0.3)
        mpc = _controller(path=tight, feedforward=True)
        with mock.patch.object(mpc_controller, 'minimize',
                               return_value=_result(np.zeros(6))) as spy:
            mpc.get_control(np.array([0.3, 0.0, math.pi / 2, 1.0]))
        self.assertLessEqual(spy.call_args.args[1][0], mpc.control_max[0])


if __name__ == '__main__':
    unittest.main()
