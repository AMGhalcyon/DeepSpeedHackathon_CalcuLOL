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


def _at(core, t, speed, last=FORWARD):
    core.t_odom = t
    core.speed = speed
    core._last_cmd = last
    return core._unstick()


def _core(driven=True, **cfg):
    """A core whose car has already driven off the grid (unstick armed), unless not."""
    mpc = MPCController(vehicle_model=VehicleModel(), **lap_sim.JUDGED_MPC)
    core = driver_core.DriverCore(cfg=driver_core.DriverConfig(**cfg), mpc_controller=mpc)
    if driven:
        _at(core, 0.0, 6.0)
    return core


class UnstickTest(unittest.TestCase):
    def test_the_default_is_on(self):
        cfg = driver_core.DriverConfig()
        self.assertEqual((cfg.unstick_after, cfg.unstick_reverse_s, cfg.unstick_speed),
                         (0.5, 0.6, 1.0))

    def test_a_moving_car_is_left_alone(self):
        core = _core()
        for k in range(40):
            self.assertIsNone(_at(core, 0.025 * k, 5.0))

    def test_a_pinned_car_reverses_counter_steering_after_the_wait(self):
        # Asked for full right (-0.419): reverse at full left, so the nose swings right.
        core = _core()
        self.assertIsNone(_at(core, 10.0, 0.01))
        self.assertIsNone(_at(core, 10.45, 0.01))
        cmd = _at(core, 10.5, 0.01)
        self.assertEqual(cmd.steering_angle, core.steer_limit)
        self.assertEqual(cmd.speed, -1.0)

    def test_it_counter_steers_a_left_turn_too(self):
        core = _core()
        left = driver_core.DriveCommand(0.3, 0.4, 0.0)
        _at(core, 10.0, 0.01, last=left)
        self.assertEqual(_at(core, 10.5, 0.01, last=left).steering_angle, -core.steer_limit)

    def test_a_car_pushing_straight_reverses_straight(self):
        core = _core()
        ahead = driver_core.DriveCommand(0.02, 0.4, 0.0)
        _at(core, 10.0, 0.01, last=ahead)
        self.assertEqual(_at(core, 10.5, 0.01, last=ahead).steering_angle, 0.0)

    def test_it_keeps_reversing_for_the_reverse_time_then_drives(self):
        core = _core()
        _at(core, 10.0, 0.01)
        first = _at(core, 10.5, 0.01)
        self.assertEqual(_at(core, 10.8, 0.6, last=first), first)   # the same command
        self.assertIsNone(_at(core, 11.15, 0.6, last=first))
        # Pinned again: another full wait, then a longer reverse (x1.5).
        self.assertIsNone(_at(core, 11.2, 0.01))
        self.assertIsNone(_at(core, 11.6, 0.01))
        self.assertEqual(_at(core, 11.7, 0.01).speed, -1.0)
        self.assertEqual(_at(core, 12.55, 0.6, last=first).speed, -1.0)  # 0.9 s
        self.assertIsNone(_at(core, 12.65, 0.6, last=first))

    def test_repeats_grow_up_to_the_cap(self):
        core = _core()
        t, lengths = 10.0, []
        for _ in range(6):
            _at(core, t, 0.01)
            cmd = _at(core, t + 0.5, 0.01)
            start, t = t + 0.5, t + 0.5
            while _at(core, t + 0.05, 0.6, last=cmd) is not None:
                t += 0.05
            t += 0.05
            lengths.append(round(t - start, 2))
        self.assertEqual(lengths[0], 0.6)
        self.assertTrue(all(b >= a for a, b in zip(lengths, lengths[1:])), lengths)
        self.assertLessEqual(max(lengths), driver_core.UNSTICK_REVERSE_MAX_S + 0.05)

    def test_a_pin_long_after_the_last_starts_over(self):
        core = _core()
        _at(core, 10.0, 0.01)
        cmd = _at(core, 10.5, 0.01)
        _at(core, 11.15, 5.0, last=cmd)                        # driving again
        _at(core, 20.0, 0.01)                                  # 9 s later
        _at(core, 20.5, 0.01)
        self.assertEqual(core._pins, 1)
        self.assertEqual(_at(core, 21.05, 0.6, last=cmd).speed, -1.0)
        self.assertIsNone(_at(core, 21.15, 0.6, last=cmd))      # 0.6 s, not 0.9

    def test_standing_still_on_purpose_is_not_stuck(self):
        core = _core()
        still = driver_core.DriveCommand(0.0, 0.0, 0.0)
        for k in range(60):
            self.assertIsNone(_at(core, 0.025 * k, 0.0, last=still))

    def test_the_launch_from_the_grid_is_not_stuck(self):
        core = _core(driven=False)
        self.assertIsNone(_at(core, 0.0, 0.0))
        self.assertIsNone(_at(core, 0.1, 0.9))       # 9.51 m/s^2 off the line
        self.assertIsNone(_at(core, 0.6, 3.0))

    def test_off_never_reverses(self):
        core = _core(unstick_after=0.0)
        for k in range(80):
            self.assertIsNone(_at(core, 0.025 * k, 0.0))

    def test_a_car_creeping_along_a_wall_is_pinned(self):
        # Judged r7-r9: full lock, full throttle, 0.37-0.5 m/s along the wall.
        core = _core()
        self.assertIsNone(_at(core, 10.0, 0.45))
        self.assertEqual(_at(core, 10.5, 0.45).speed, -1.0)

    def test_an_unknown_speed_is_not_pinned(self):
        core = _core()
        for k in range(40):
            self.assertIsNone(_at(core, 10.0 + 0.025 * k, float('nan')))

    def test_a_sim_reset_forgets_the_stuck_timer(self):
        core = _core()
        _at(core, 10.0, 0.01)
        core._reset_unstick()
        self.assertIsNone(_at(core, 10.5, 0.01))

    def test_a_car_that_has_not_driven_yet_never_reverses(self):
        # Judged S-r7: the first commands had not reached the car; it sat on the
        # grid asking to move and was reversed off it.
        core = _core(driven=False)
        for k in range(80):                                    # 2 s still, pushing
            self.assertIsNone(_at(core, 0.025 * k, 0.0))

    def test_it_arms_once_the_car_has_driven(self):
        core = _core(driven=False)
        _at(core, 1.0, 2.5)                                    # off the grid
        _at(core, 2.0, 0.01)
        self.assertEqual(_at(core, 2.5, 0.01).speed, -1.0)

    def test_a_sim_reset_disarms(self):
        core = _core()
        core._reset_unstick()
        _at(core, 10.0, 0.01)
        self.assertIsNone(_at(core, 10.5, 0.01))

    def test_a_sim_reset_forgets_the_repeats(self):
        core = _core()
        _at(core, 10.0, 0.01)
        _at(core, 10.5, 0.01)
        core._reset_unstick()
        self.assertEqual((core._pins, core._reverse_ended, core._reverse_until), (0, None, None))


class StepTest(unittest.TestCase):
    """Through `step`, the path the node and the harness call."""

    def _tick(self, core, t, speed):
        core.set_scan([5.0] * 1080, -2.35, 0.0043633, t_sim=t)
        core.set_odom(*core.reference_path[0], 0.0, speed, t_sim=t)
        cmd = core.step(t_sim=t)
        core.note_published(cmd, t_sim=t)
        return cmd

    def test_a_pinned_car_is_sent_backwards_by_step(self):
        core = _core(driven=False)
        self._tick(core, 0.025, 6.0)                            # was driving
        cmds = [self._tick(core, 0.025 * k, 0.0) for k in range(2, 31)]
        self.assertGreater(cmds[0].speed, 0.0)                  # asks to move
        self.assertEqual(cmds[-1].speed, -core.cfg.unstick_speed)
        self.assertEqual(core._cond_steer, cmds[-1].steering_angle)
        self.assertIn(abs(cmds[-1].steering_angle), (0.0, core.steer_limit))


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
