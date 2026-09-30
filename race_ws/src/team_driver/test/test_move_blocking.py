"""Step 5: move blocking, an iteration cap, and a wall-clock solve budget."""

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

JUDGED_BLOCK_MAP = [0, 1, 1, 2, 2, 2, 2, 2]
HORIZON = len(JUDGED_BLOCK_MAP)
BLOCKS = 3
DT = 0.025
STATE = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
SOLVER_ITERATION_LIMIT = 9


def _controller(**overrides):
    kwargs = dict(vehicle_model=VehicleModel(), prediction_horizon=HORIZON,
                  block_map=JUDGED_BLOCK_MAP, dt=DT)
    kwargs.update(overrides)
    mpc = MPCController(**kwargs)
    mpc.set_reference_path(np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]))
    return mpc


def _result(sequence, success=True, status=0):
    return mock.Mock(success=success, status=status, message='stub',
                     x=np.asarray(sequence, dtype=float))


def _blocks(*rows):
    return np.array(rows, dtype=float).ravel()


class BlockMapTest(unittest.TestCase):

    def test_block_map_covers_every_prediction_step(self):
        mpc = _controller()
        self.assertEqual(len(mpc.block_map), mpc.prediction_horizon)

    def test_judged_map_is_one_step_then_two_then_the_rest(self):
        np.testing.assert_array_equal(_controller().block_map, JUDGED_BLOCK_MAP)

    def test_control_horizon_is_the_number_of_blocks(self):
        self.assertEqual(_controller().control_horizon, BLOCKS)

    def test_lookahead_is_unchanged_by_blocking(self):
        mpc = _controller()
        self.assertAlmostEqual(mpc.prediction_horizon * mpc.dt, 0.2)

    def test_decision_variables_drop_from_sixteen_to_six(self):
        mpc = _controller()
        with mock.patch.object(
                mpc_controller, 'minimize',
                return_value=_result(np.zeros(BLOCKS * 2))) as spy:
            mpc.get_control(STATE)
        self.assertEqual(len(spy.call_args.args[1]), 6)
        self.assertEqual(len(spy.call_args.kwargs['bounds']), 6)

    def test_rejects_a_map_whose_length_is_not_the_prediction_horizon(self):
        with self.assertRaisesRegex(ValueError, 'block_map'):
            _controller(block_map=[0, 1, 1, 2])

    def test_rejects_a_map_that_does_not_start_at_block_zero(self):
        with self.assertRaisesRegex(ValueError, 'block_map'):
            _controller(block_map=[1, 1, 1, 2, 2, 2, 2, 2])

    def test_rejects_a_map_that_skips_a_block(self):
        with self.assertRaisesRegex(ValueError, 'block_map'):
            _controller(block_map=[0, 0, 2, 2, 2, 2, 2, 2])

    def test_rejects_a_map_that_goes_backwards(self):
        with self.assertRaisesRegex(ValueError, 'block_map'):
            _controller(block_map=[0, 1, 2, 1, 2, 2, 2, 2])

    def test_rejects_a_control_horizon_that_contradicts_the_map(self):
        with self.assertRaisesRegex(ValueError, 'control_horizon'):
            _controller(control_horizon=5)

    def test_accepts_a_control_horizon_that_agrees_with_the_map(self):
        self.assertEqual(_controller(control_horizon=BLOCKS).control_horizon, BLOCKS)

    def test_without_a_map_control_horizon_holds_the_last_block(self):
        mpc = _controller(block_map=None, control_horizon=3)
        np.testing.assert_array_equal(mpc.block_map, [0, 1, 2, 2, 2, 2, 2, 2])

    def test_without_a_map_or_a_control_horizon_nothing_is_blocked(self):
        mpc = _controller(block_map=None)
        np.testing.assert_array_equal(mpc.block_map, np.arange(HORIZON))


class PredictionTest(unittest.TestCase):

    def test_each_step_applies_the_control_of_its_block(self):
        mpc = _controller()
        blocks = _blocks([0.1, 1.0], [0.2, 2.0], [0.3, 3.0])
        applied = []

        def record(state, control, dt):
            applied.append(np.array(control))
            return state

        # Step 6: the per-step dynamics are the actuated step (wheel as state).
        with mock.patch.object(mpc.vehicle_model, 'actuated_step',
                               side_effect=record):
            mpc._predict_trajectory(np.append(STATE, 0.0), blocks)

        expected = blocks.reshape(BLOCKS, 2)[JUDGED_BLOCK_MAP]
        np.testing.assert_allclose(np.array(applied), expected)


class IterationCapTest(unittest.TestCase):

    def test_default_cap_is_twenty_five_iterations(self):
        mpc = _controller()
        with mock.patch.object(
                mpc_controller, 'minimize',
                return_value=_result(np.zeros(BLOCKS * 2))) as spy:
            mpc.get_control(STATE)
        self.assertEqual(spy.call_args.kwargs['options']['maxiter'], 25)

    def test_cap_is_configurable(self):
        mpc = _controller(maxiter=7)
        with mock.patch.object(
                mpc_controller, 'minimize',
                return_value=_result(np.zeros(BLOCKS * 2))) as spy:
            mpc.get_control(STATE)
        self.assertEqual(spy.call_args.kwargs['options']['maxiter'], 7)

    def test_a_capped_solve_still_applies_its_partial_result(self):
        mpc = _controller()
        partial = _blocks([0.05, 0.5], [0.06, 0.6], [0.07, 0.7])
        result = _result(partial, success=False, status=SOLVER_ITERATION_LIMIT)
        with mock.patch.object(mpc_controller, 'minimize', return_value=result):
            control, _ = mpc.get_control(STATE)
        np.testing.assert_allclose(control, [0.05, 0.5])
        np.testing.assert_allclose(mpc._u_prev, partial)
        self.assertIsNone(mpc.last_fallback)


class WarmStartTest(unittest.TestCase):

    def _second_solve_guess(self, first):
        mpc = _controller()
        with mock.patch.object(
                mpc_controller, 'minimize', return_value=_result(first)):
            mpc.get_control(STATE)
        with mock.patch.object(
                mpc_controller, 'minimize', return_value=_result(first)) as spy:
            mpc.get_control(STATE)
        return spy.call_args.args[1]

    def test_warm_start_shifts_one_time_step_not_one_block(self):
        a, b, c = [0.1, 1.0], [0.2, 2.0], [0.3, 3.0]
        guess = self._second_solve_guess(_blocks(a, b, c))
        # Per-step sequence is a b b c c c c c; one step on it is b b c c c c c c.
        mean_bc = (np.array(b) + np.array(c)) / 2.0
        np.testing.assert_allclose(guess, _blocks(b, mean_bc, c))

    def test_constant_solution_is_its_own_warm_start(self):
        u = [0.1, 1.0]
        guess = self._second_solve_guess(_blocks(u, u, u))
        np.testing.assert_allclose(guess, _blocks(u, u, u))


class FallbackTest(unittest.TestCase):

    def _primed(self, **overrides):
        """A controller that has applied a known solution once."""
        mpc = _controller(**overrides)
        self.a, self.b, self.c = [0.1, 1.0], [0.2, 2.0], [0.3, 3.0]
        with mock.patch.object(
                mpc_controller, 'minimize',
                return_value=_result(_blocks(self.a, self.b, self.c))):
            mpc.get_control(STATE)
        return mpc

    def test_budget_exhaustion_returns_the_warm_start_not_the_last_command(self):
        mpc = self._primed(solve_budget_ms=0.0)
        control, _ = mpc.get_control(STATE)        # real SLSQP, budget already gone
        np.testing.assert_allclose(control, self.b)
        self.assertFalse(np.allclose(control, self.a))
        self.assertEqual(mpc.last_fallback, 'budget')

    def test_budget_fallback_advances_the_memory_one_step(self):
        mpc = self._primed(solve_budget_ms=0.0)
        mpc.get_control(STATE)
        mean_bc = (np.array(self.b) + np.array(self.c)) / 2.0
        np.testing.assert_allclose(
            mpc._u_prev, _blocks(self.b, mean_bc, self.c))
        np.testing.assert_allclose(mpc.prev_control, self.b)

    def test_fallback_returns_a_copy(self):
        mpc = self._primed(solve_budget_ms=0.0)
        control, _ = mpc.get_control(STATE)
        control[0] += 1.0
        np.testing.assert_allclose(mpc.prev_control, self.b)
        self.assertAlmostEqual(mpc._u_prev[0], self.b[0])

    def test_budget_exhaustion_is_counted(self):
        mpc = self._primed(solve_budget_ms=0.0)
        mpc.get_control(STATE)
        mpc.get_control(STATE)
        self.assertEqual(mpc.budget_exceeded_count, 2)

    def test_a_roomy_budget_does_not_trip(self):
        mpc = _controller(solve_budget_ms=10_000.0)
        mpc.get_control(STATE)
        self.assertIsNone(mpc.last_fallback)
        self.assertEqual(mpc.budget_exceeded_count, 0)

    def test_a_hard_solver_failure_returns_the_warm_start(self):
        mpc = self._primed()
        failed = _result(np.full(BLOCKS * 2, 9.0), success=False, status=4)
        with mock.patch.object(mpc_controller, 'minimize', return_value=failed):
            control, _ = mpc.get_control(STATE)
        np.testing.assert_allclose(control, self.b)
        self.assertEqual(mpc.last_fallback, 'solver')

    def test_a_non_finite_result_is_not_applied(self):
        mpc = self._primed()
        bad = _result(np.full(BLOCKS * 2, np.nan))
        with mock.patch.object(mpc_controller, 'minimize', return_value=bad):
            control, _ = mpc.get_control(STATE)
        self.assertTrue(np.all(np.isfinite(control)))
        np.testing.assert_allclose(control, self.b)

    def test_an_optimizer_exception_returns_the_warm_start(self):
        mpc = self._primed()
        with mock.patch.object(mpc_controller, 'minimize',
                               side_effect=RuntimeError('boom')):
            control, _ = mpc.get_control(STATE)
        np.testing.assert_allclose(control, self.b)
        self.assertEqual(mpc.last_fallback, 'error')

    def test_get_control_reports_the_fallback(self):
        mpc = self._primed(solve_budget_ms=0.0)
        _, diagnostics = mpc.get_control(STATE)
        self.assertEqual(diagnostics['fallback'], 'budget')


if __name__ == '__main__':
    unittest.main()
