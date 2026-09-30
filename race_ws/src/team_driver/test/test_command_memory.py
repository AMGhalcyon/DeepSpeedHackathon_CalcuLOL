"""Step 4: the MPC remembers its own last command; no caller can inject one."""

import inspect
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

HORIZON = 4
STATE = np.array([0.0, 0.0, 0.0, 1.0, 0.0])


def _controller():
    mpc = MPCController(
        vehicle_model=VehicleModel(),
        prediction_horizon=HORIZON, control_horizon=HORIZON, dt=0.025)
    mpc.set_reference_path(np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]))
    return mpc


def _fake_result(sequence):
    return mock.Mock(success=True, x=np.asarray(sequence, dtype=float))


class ApiTest(unittest.TestCase):

    def test_solve_takes_only_the_state(self):
        params = list(inspect.signature(MPCController.solve).parameters)
        self.assertEqual(params, ['self', 'current_state'])

    def test_get_control_takes_only_the_state(self):
        params = list(inspect.signature(MPCController.get_control).parameters)
        self.assertEqual(params, ['self', 'current_state'])

    def test_a_caller_cannot_inject_a_previous_command(self):
        with self.assertRaises(TypeError):
            _controller().get_control(STATE, np.array([0.3, 1.0]))


class CommandMemoryTest(unittest.TestCase):

    def test_prev_control_is_the_controllers_own_output(self):
        mpc = _controller()
        control, _ = mpc.get_control(STATE)
        np.testing.assert_allclose(mpc.prev_control, control)

    def test_prev_control_is_a_copy_not_a_view_of_the_result(self):
        mpc = _controller()
        control, _ = mpc.get_control(STATE)
        control[0] += 1.0
        self.assertNotEqual(mpc.prev_control[0], control[0])

    def test_first_solve_starts_from_zero(self):
        mpc = _controller()
        with mock.patch.object(
                mpc_controller, 'minimize',
                return_value=_fake_result(np.zeros(HORIZON * 2))) as spy:
            mpc.get_control(STATE)
        np.testing.assert_array_equal(spy.call_args.args[1], np.zeros(HORIZON * 2))

    def test_second_solve_is_warm_started_one_step_shifted(self):
        mpc = _controller()
        first = np.arange(HORIZON * 2, dtype=float) / 10.0
        with mock.patch.object(
                mpc_controller, 'minimize', return_value=_fake_result(first)):
            mpc.get_control(STATE)
        with mock.patch.object(
                mpc_controller, 'minimize',
                return_value=_fake_result(first)) as spy:
            mpc.get_control(STATE)
        seq = first.reshape(HORIZON, 2)
        expected = np.vstack([seq[1:], seq[-1:]]).ravel()
        np.testing.assert_allclose(spy.call_args.args[1], expected)

    def test_failed_solve_falls_back_to_the_shifted_warm_start(self):
        # Step 5: a failed solve continues the last plan one step on, rather
        # than repeating the last command.
        mpc = _controller()
        good = np.arange(HORIZON * 2, dtype=float) / 10.0
        with mock.patch.object(
                mpc_controller, 'minimize', return_value=_fake_result(good)):
            mpc.get_control(STATE)
        failed = mock.Mock(success=False, status=4, x=np.full(HORIZON * 2, 9.0),
                           message='boom')
        with mock.patch.object(mpc_controller, 'minimize', return_value=failed):
            control, _ = mpc.get_control(STATE)
        second_step = good.reshape(HORIZON, 2)[1]
        np.testing.assert_allclose(control, second_step)
        np.testing.assert_allclose(mpc.prev_control, second_step)


if __name__ == '__main__':
    unittest.main()
