"""The harness can run a different set of MPC weights without editing code.

Step 9 sweeps `r_delta_steer`, `q_pos` and `q_yaw`. Each sweep point has to be
reproducible from one command line, so `lap_sim.py --mpc KEY=VALUE` overrides a
weight of JUDGED_MPC for that run only. Only scalar weights and the solve budget
are open: the horizon, block map and dt are structure, and changing them here
would break the node's startup assertions without the harness noticing.
"""

import json
import os
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'team_driver'))

import lap_sim  # noqa: E402


class ParseOverridesTest(unittest.TestCase):

    def assertRejected(self, argv):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            lap_sim.parse_config(argv)

    def test_no_flag_means_no_overrides(self):
        self.assertEqual(lap_sim.parse_config([]).mpc_overrides, ())

    def test_one_flag_sets_one_weight(self):
        cfg = lap_sim.parse_config(['--mpc', 'r_delta_steer=0.3'])
        self.assertEqual(cfg.mpc_overrides, (('r_delta_steer', 0.3),))

    def test_repeated_flags_accumulate_in_order(self):
        cfg = lap_sim.parse_config(['--mpc', 'q_pos=4', '--mpc', 'q_yaw=1.5'])
        self.assertEqual(cfg.mpc_overrides, (('q_pos', 4.0), ('q_yaw', 1.5)))

    def test_the_solve_budget_can_be_lifted(self):
        cfg = lap_sim.parse_config(['--mpc', 'solve_budget_ms=1e6'])
        self.assertEqual(cfg.mpc_overrides, (('solve_budget_ms', 1e6),))

    def test_an_unknown_key_is_rejected(self):
        self.assertRejected(['--mpc', 'q_banana=1'])

    def test_structural_keys_are_rejected(self):
        for key in ('prediction_horizon', 'control_horizon', 'dt', 'block_map', 'maxiter'):
            with self.subTest(key=key):
                self.assertRejected(['--mpc', f'{key}=3'])

    def test_a_value_that_is_not_a_number_is_rejected(self):
        self.assertRejected(['--mpc', 'q_pos=lots'])

    def test_a_missing_equals_sign_is_rejected(self):
        self.assertRejected(['--mpc', 'q_pos'])

    def test_a_negative_weight_is_rejected(self):
        self.assertRejected(['--mpc', 'r_delta_steer=-0.1'])

    def test_a_non_finite_weight_is_rejected(self):
        self.assertRejected(['--mpc', 'q_pos=nan'])
        self.assertRejected(['--mpc', 'q_pos=inf'])

    def test_a_zero_weight_is_allowed(self):
        cfg = lap_sim.parse_config(['--mpc', 'q_yaw=0'])
        self.assertEqual(cfg.mpc_overrides, (('q_yaw', 0.0),))

    def test_a_zero_solve_budget_is_rejected(self):
        # The deadline would already have passed: every solve aborts at once.
        self.assertRejected(['--mpc', 'solve_budget_ms=0'])


class JsonRecordTest(unittest.TestCase):

    def test_the_record_holds_the_weights_actually_used(self):
        cfg = lap_sim.HarnessConfig(mpc_overrides=(('q_pos', 4.0),))
        record = lap_sim.json_record(cfg, {'laps': 0})
        self.assertEqual(record['mpc'], lap_sim.mpc_kwargs(cfg))
        self.assertEqual(record['config']['mpc_overrides'], (('q_pos', 4.0),))
        self.assertEqual(record['result'], {'laps': 0})

    def test_the_record_round_trips_through_json(self):
        cfg = lap_sim.HarnessConfig(mpc_overrides=(('q_pos', 4.0),))
        record = lap_sim.json_record(cfg, {'laps': 0})
        self.assertEqual(json.loads(json.dumps(record))['mpc']['q_pos'], 4.0)


class MpcKwargsTest(unittest.TestCase):

    def test_no_overrides_is_the_judged_set(self):
        self.assertEqual(lap_sim.mpc_kwargs(lap_sim.HarnessConfig()), lap_sim.JUDGED_MPC)

    def test_an_override_replaces_only_its_key(self):
        cfg = lap_sim.HarnessConfig(mpc_overrides=(('q_pos', 4.0),))
        kwargs = lap_sim.mpc_kwargs(cfg)
        self.assertEqual(kwargs['q_pos'], 4.0)
        self.assertEqual({k: v for k, v in kwargs.items() if k != 'q_pos'},
                         {k: v for k, v in lap_sim.JUDGED_MPC.items() if k != 'q_pos'})

    def test_the_judged_set_is_not_mutated(self):
        before = dict(lap_sim.JUDGED_MPC)
        lap_sim.mpc_kwargs(lap_sim.HarnessConfig(mpc_overrides=(('q_pos', 9.0),)))
        self.assertEqual(lap_sim.JUDGED_MPC, before)


class _Built(Exception):
    pass


class RunUsesOverridesTest(unittest.TestCase):

    def test_run_builds_the_controller_with_the_overrides(self):
        seen = {}

        def capture(**kwargs):
            seen.update(kwargs)
            raise _Built

        cfg = lap_sim.HarnessConfig(laps=1, quiet=True,
                                    mpc_overrides=(('r_delta_steer', 0.2),))
        with mock.patch.object(lap_sim, 'MPCController', side_effect=capture), \
                self.assertRaises(_Built):
            lap_sim.run(cfg)
        self.assertEqual(seen['r_delta_steer'], 0.2)
        self.assertEqual(seen['q_pos'], lap_sim.JUDGED_MPC['q_pos'])


if __name__ == '__main__':
    unittest.main()
