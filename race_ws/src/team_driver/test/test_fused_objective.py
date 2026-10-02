"""Phase 2 Step 3: the solve budget, kept by a fused objective and gradient.

SLSQP spent ~115 us per cost evaluation, of which the rollout was a third: the
rest was numpy temporaries in the cost and scipy's finite-difference
bookkeeping. `_objective` / `_gradient` compute the same cost on Python floats
and the same forward differences (scipy's step, scipy's bound handling), and a
perturbed rollout restarts at the first step its block touches. Same optimum,
about a third of the time.
"""

import math
import os
import sys
import time
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

from mpc_controller import MPCController, SolveBudgetExceeded  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

BLOCK_MAP = [0, 1, 1, 2, 2, 2, 2, 2]
HORIZON = len(BLOCK_MAP)
LONG = [0.025, 0.025, 0.05, 0.05, 0.075, 0.075, 0.1, 0.1]
SLSQP_EPS = 1.4901161193847656e-08
TIMING_REPEATS = 5


def _circle(radius=4.0, n=400):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _armed(model, schedule, state, heading_offset=0.0):
    """A controller with its reference window built for `state`, as solve() does."""
    mpc = MPCController(vehicle_model=VehicleModel(model_type=model),
                        prediction_horizon=HORIZON, block_map=BLOCK_MAP, dt=0.025,
                        dt_schedule=schedule, r_delta_steer=3.0, solve_budget_ms=1e6)
    path = _circle()
    mpc.set_reference_path(path, np.full(len(path), 3.0))
    ref = mpc._reference_window(0.5, HORIZON + 1, car_speed=state[3]).copy()
    ref[:, 2] += heading_offset
    mpc.prev_control = np.array([0.05, 0.3])
    mpc._prepare_objective(np.asarray(state, dtype=float), ref)
    return mpc, ref


CASES = [
    ('KS', None, [4.0, 0.1, math.pi / 2, 3.0, 0.0, 0.05]),
    ('KS', LONG, [4.0, 0.1, math.pi / 2, 3.0, 0.0, 0.05]),
    ('ST', LONG, [4.0, 0.1, math.pi / 2, 5.0, 0.04, 0.08, 1.1]),
    ('ST', None, [4.0, 0.1, math.pi / 2, 0.3, 0.0, 0.08, 0.0]),   # below the switch
]


class ScalarStepTest(unittest.TestCase):

    def test_kinematic_scalar_step_is_the_actuated_step_to_the_bit(self):
        model = VehicleModel()
        state = np.array([1.0, 2.0, 0.3, 3.0, 0.0, 0.05])
        for h in (0.025, 0.075):
            a = model.actuated_step(state, (0.2, 1.5), h)
            b = model.scalar_step(tuple(state), 0.2, 1.5, h)
            self.assertEqual(tuple(a), b)

    def test_single_track_scalar_step_is_the_actuated_step(self):
        model = VehicleModel(model_type='ST')
        state = np.array([1.0, 2.0, 0.3, 5.0, 0.03, 0.05, 0.9])
        a = model.actuated_step(state, (0.2, 1.5), 0.05)
        b = model.scalar_step(tuple(state), 0.2, 1.5, 0.05)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)


class ObjectiveTest(unittest.TestCase):

    def test_the_objective_is_the_cost_function(self):
        rng = np.random.default_rng(0)
        for model, schedule, state in CASES:
            with self.subTest(model=model, schedule=schedule is not None, v=state[3]):
                mpc, ref = _armed(model, schedule, state)
                for _ in range(5):
                    u = rng.uniform([-0.3, -2.0] * 3, [0.3, 5.0] * 3)
                    expected = mpc._cost_function(u, np.asarray(state), ref, ref[:, 3])
                    self.assertAlmostEqual(mpc._objective(u), expected,
                                           delta=1e-12 * max(1.0, expected))

    def test_heading_errors_wrap_across_the_branch_cut(self):
        state = [4.0, 0.1, math.pi - 0.01, 3.0, 0.0, 0.0]
        mpc, ref = _armed('KS', None, state)
        ref[:, 2] = -math.pi + 0.01          # 0.02 rad away, the long way round 2 pi
        mpc._prepare_objective(np.asarray(state), ref)
        u = np.zeros(6)
        expected = mpc._cost_function(u, np.asarray(state), ref, ref[:, 3])
        self.assertAlmostEqual(mpc._objective(u), expected, delta=1e-9)


class GradientTest(unittest.TestCase):

    def _scipy_gradient(self, mpc, ref, state, u):
        lo = np.tile(mpc.control_min, mpc.control_horizon)
        hi = np.tile(mpc.control_max, mpc.control_horizon)
        return approx_derivative(
            lambda z: mpc._cost_function(z, np.asarray(state), ref, ref[:, 3]),
            u, method='2-point', abs_step=SLSQP_EPS, bounds=(lo, hi))

    def test_the_gradient_is_scipys_forward_difference(self):
        rng = np.random.default_rng(1)
        for model, schedule, state in CASES:
            with self.subTest(model=model, schedule=schedule is not None, v=state[3]):
                mpc, ref = _armed(model, schedule, state)
                u = rng.uniform([-0.3, -2.0] * 3, [0.3, 5.0] * 3)
                np.testing.assert_allclose(mpc._gradient(u),
                                           self._scipy_gradient(mpc, ref, state, u),
                                           rtol=1e-5, atol=1e-5)

    def test_at_the_upper_bound_the_step_goes_backwards(self):
        state = CASES[0][2]
        mpc, ref = _armed('KS', None, state)
        u = np.array([mpc.control_max[0], 1.0, 0.1, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(mpc._gradient(u),
                                   self._scipy_gradient(mpc, ref, state, u),
                                   rtol=1e-5, atol=1e-5)


class SolveTest(unittest.TestCase):

    def test_the_fused_solve_lands_where_the_scipy_solve_does(self):
        for model, schedule, state in CASES[:3]:
            with self.subTest(model=model, schedule=schedule is not None):
                kwargs = dict(vehicle_model=VehicleModel(model_type=model),
                              prediction_horizon=HORIZON, block_map=BLOCK_MAP, dt=0.025,
                              dt_schedule=schedule, r_delta_steer=3.0,
                              solve_budget_ms=1e6, maxiter=200)
                fused = MPCController(**kwargs)
                reference = MPCController(**kwargs)
                reference.fused_objective = False
                for mpc in (fused, reference):
                    mpc.set_reference_path(_circle(), np.full(400, 3.0))
                a, _ = fused.get_control(np.array(state))
                b, _ = reference.get_control(np.array(state))
                np.testing.assert_allclose(a, b, atol=2e-4)

    def test_the_fused_solve_is_faster(self):
        # Measured ~5x. Best of several repetitions, so a load spike on the host
        # can only slow a repetition down, not flip the comparison.
        best = {}
        for fused in (True, False):
            mpc = MPCController(vehicle_model=VehicleModel(model_type='ST'),
                                prediction_horizon=HORIZON, block_map=BLOCK_MAP,
                                dt=0.025, dt_schedule=LONG, solve_budget_ms=1e6)
            mpc.fused_objective = fused
            mpc.set_reference_path(_circle(), np.full(400, 3.0))
            state = np.array(CASES[2][2])
            runs = []
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                for _ in range(5):
                    mpc.get_control(state)
                runs.append(time.perf_counter() - t0)
            best[fused] = min(runs)
        self.assertLess(best[True], 0.6 * best[False])

    def test_a_gradient_at_a_new_point_recomputes_the_base(self):
        # SLSQP normally asks for the gradient where it just evaluated; when it
        # does not, the recorded base trajectory must not be reused.
        mpc, ref = _armed(*CASES[2])
        state = CASES[2][2]
        mpc._objective(np.zeros(6))
        u = np.array([0.1, 1.0, 0.05, 0.0, -0.1, 2.0])
        np.testing.assert_allclose(mpc._gradient(u),
                                   GradientTest()._scipy_gradient(mpc, ref, state, u),
                                   rtol=1e-5, atol=1e-5)

    def test_the_gradient_is_right_with_the_acceleration_at_its_bound(self):
        mpc, ref = _armed(*CASES[2])
        u = np.array([0.05, mpc.control_max[1], 0.05, mpc.control_min[1], 0.0, 1.0])
        np.testing.assert_allclose(mpc._gradient(u),
                                   GradientTest()._scipy_gradient(mpc, ref, CASES[2][2], u),
                                   rtol=1e-5, atol=1e-5)


class BudgetTest(unittest.TestCase):

    def test_the_objective_and_gradient_honour_the_deadline(self):
        mpc, _ = _armed(*CASES[0])
        mpc._deadline = time.perf_counter() - 1.0
        with self.assertRaises(SolveBudgetExceeded):
            mpc._objective(np.zeros(6))
        with self.assertRaises(SolveBudgetExceeded):
            mpc._gradient(np.zeros(6))

    def test_a_spent_budget_still_falls_back(self):
        mpc = MPCController(vehicle_model=VehicleModel(), prediction_horizon=HORIZON,
                            block_map=BLOCK_MAP, dt=0.025, solve_budget_ms=0.0)
        mpc.set_reference_path(_circle(), np.full(400, 3.0))
        mpc.get_control(np.array(CASES[0][2]))
        self.assertEqual(mpc.last_fallback, 'budget')


if __name__ == '__main__':
    unittest.main()
