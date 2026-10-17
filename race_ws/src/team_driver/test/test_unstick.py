"""Sub-9 phase 2 Step 2: back off a wall instead of pushing into it until disqualified.

Judged r1 (20261017T195113Z): after one hit at s 44 the car sat against the wall
at v ~0 with full lock and full throttle forward for 4 s -- 11 collisions, DQ.
The driver never asks for reverse. Stuck = under UNSTICK_SPEED while its last
command asked to move, for `unstick_after`; then reverse straight for
`unstick_reverse_s` at `unstick_speed`, then drive again.
"""

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

FORWARD = driver_core.DriveCommand(-0.419, 0.4, 0.0)


def _core(**cfg):
    mpc = MPCController(vehicle_model=VehicleModel(), **lap_sim.JUDGED_MPC)
    return driver_core.DriverCore(cfg=driver_core.DriverConfig(**cfg), mpc_controller=mpc)


def _at(core, t, speed, last=FORWARD):
    core.t_odom = t
    core.speed = speed
    core._last_cmd = last
    return core._unstick()


class UnstickTest(unittest.TestCase):
    def test_the_default_is_on(self):
        cfg = driver_core.DriverConfig()
        self.assertEqual((cfg.unstick_after, cfg.unstick_reverse_s, cfg.unstick_speed),
                         (0.5, 0.6, 1.0))

    def test_a_moving_car_is_left_alone(self):
        core = _core()
        for k in range(40):
            self.assertIsNone(_at(core, 0.025 * k, 5.0))

    def test_a_pinned_car_reverses_straight_after_the_wait(self):
        core = _core()
        self.assertIsNone(_at(core, 10.0, 0.01))
        self.assertIsNone(_at(core, 10.45, 0.01))
        cmd = _at(core, 10.5, 0.01)
        self.assertEqual(cmd.steering_angle, 0.0)
        self.assertEqual(cmd.speed, -1.0)

    def test_it_keeps_reversing_for_the_reverse_time_then_drives(self):
        core = _core()
        _at(core, 10.0, 0.01)
        _at(core, 10.5, 0.01)
        back = driver_core.DriveCommand(0.0, -1.0, 0.0)
        self.assertEqual(_at(core, 10.8, 0.6, last=back).speed, -1.0)
        self.assertIsNone(_at(core, 11.15, 0.6, last=back))
        # Pinned again: another full wait before the next reverse.
        self.assertIsNone(_at(core, 11.2, 0.01))
        self.assertIsNone(_at(core, 11.6, 0.01))
        self.assertEqual(_at(core, 11.7, 0.01).speed, -1.0)

    def test_standing_still_on_purpose_is_not_stuck(self):
        core = _core()
        still = driver_core.DriveCommand(0.0, 0.0, 0.0)
        for k in range(60):
            self.assertIsNone(_at(core, 0.025 * k, 0.0, last=still))

    def test_the_launch_from_the_grid_is_not_stuck(self):
        core = _core()
        self.assertIsNone(_at(core, 0.0, 0.0))
        self.assertIsNone(_at(core, 0.1, 0.9))       # 9.51 m/s^2 off the line
        self.assertIsNone(_at(core, 0.6, 3.0))

    def test_off_never_reverses(self):
        core = _core(unstick_after=0.0)
        for k in range(80):
            self.assertIsNone(_at(core, 0.025 * k, 0.0))

    def test_a_sim_reset_forgets_the_stuck_timer(self):
        core = _core()
        _at(core, 10.0, 0.01)
        core._reset_unstick()
        self.assertIsNone(_at(core, 10.5, 0.01))


class StepTest(unittest.TestCase):
    """Through `step`, the path the node and the harness call."""

    def _tick(self, core, t, speed):
        core.set_scan([5.0] * 1080, -2.35, 0.0043633, t_sim=t)
        core.set_odom(*core.reference_path[0], 0.0, speed, t_sim=t)
        cmd = core.step(t_sim=t)
        core.note_published(cmd, t_sim=t)
        return cmd

    def test_a_pinned_car_is_sent_backwards_by_step(self):
        core = _core()
        cmds = [self._tick(core, 0.025 * k, 0.0) for k in range(1, 30)]
        self.assertGreater(cmds[0].speed, 0.0)                  # asks to move
        self.assertEqual(cmds[-1].speed, -core.cfg.unstick_speed)
        self.assertEqual(core._cond_steer, 0.0)


class ValidationTest(unittest.TestCase):
    def test_bad_values_are_refused(self):
        for over in (dict(unstick_after=-0.1), dict(unstick_reverse_s=0.0),
                     dict(unstick_speed=0.0), dict(unstick_speed=-1.0)):
            with self.subTest(**over):
                with self.assertRaises(ValueError):
                    driver_params.validate({**driver_params.DEFAULTS, **over})
                with self.assertRaises(ValueError):
                    driver_core.DriverConfig(**over)


if __name__ == '__main__':
    unittest.main()
