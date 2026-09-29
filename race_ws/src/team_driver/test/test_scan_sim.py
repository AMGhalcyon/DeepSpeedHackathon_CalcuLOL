"""Tests pinning `scan_sim` to the judge's sensor, walls and collision rule.

Two separable claims are being defended here.

The first is that the geometry is the judge's, not a lookalike: walls come from
its own `extract_walls`, the sensor sits where `_lidar_pose_from_cog` puts it,
the body box sits where `_collision_pose_from_base` puts it, and the
ray/segment kernel agrees with hand-computed analytic answers. A map loaded
with the wrong row order, or a sensor placed at the CoG instead of 0.275 m
ahead of the rear axle, produces a plausible-looking scan that is quietly
wrong everywhere -- so these are pinned against arithmetic, not against
themselves.

The second is that the broad-phase cull is exact rather than a speed/accuracy
trade: no segment farther from the sensor than the cull radius can be hit
within it, so culling must change nothing at all below that distance.
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
import scan_sim  # noqa: E402
from scan_sim import (CollisionCounter, ContactResolver,  # noqa: E402
                      ScanSimulator, load_map, ray_segment_range)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
MAP_YAML = os.path.join(REPO, 'maps', 'icra26.yaml')

# maps/tracks.yaml: on the bottom straight, facing along +x. Same pose the
# reference-wrap tests use, and the pose the referee spawns from.
START_BASE_LINK = (-1.68, -0.01, 0.0)
# The driver clips every range to this (DriverConfig.max_range), so a cull
# radius above it cannot change any value the controller can observe.
DRIVER_CLIP_M = 8.0

_TRACK = None


def _track():
    """The real circuit, loaded once -- wall extraction is the slow part."""
    global _TRACK
    if _TRACK is None:
        _TRACK = load_map(MAP_YAML)
    return _TRACK


def _start_cog():
    """CoG pose for the spawn, via the plant's own base_link conversion."""
    plant = plant_sim.GymPlant()
    plant.reset(START_BASE_LINK)
    return (float(plant.state[plant_sim.X]),
            float(plant.state[plant_sim.Y]),
            float(plant.state[plant_sim.PSI]))


class TestRaySegmentKernel(unittest.TestCase):
    """The numpy port of `lidar/kernels.py`, against analytic answers."""

    def test_perpendicular_hit(self):
        """A wall across the path at x = 1 is exactly 1 m away."""
        seg_a = np.array([[1.0, -1.0]])
        seg_b = np.array([[1.0, 1.0]])
        origin = np.array([0.0, 0.0])
        direction = np.array([[1.0, 0.0]])
        got = ray_segment_range(origin, direction, seg_a, seg_b, 25.0)
        self.assertAlmostEqual(float(got[0]), 1.0, places=9)

    def test_diagonal_hit(self):
        """45 degrees into the same wall is sqrt(2) away."""
        seg_a = np.array([[1.0, -2.0]])
        seg_b = np.array([[1.0, 2.0]])
        origin = np.array([0.0, 0.0])
        direction = np.array([[math.cos(math.pi / 4), math.sin(math.pi / 4)]])
        got = ray_segment_range(origin, direction, seg_a, seg_b, 25.0)
        self.assertAlmostEqual(float(got[0]), math.sqrt(2.0), places=9)

    def test_ray_pointing_away_does_not_hit(self):
        """Negative ray fractions are behind the sensor and must be rejected."""
        seg_a = np.array([[1.0, -1.0]])
        seg_b = np.array([[1.0, 1.0]])
        got = ray_segment_range(np.array([0.0, 0.0]), np.array([[-1.0, 0.0]]),
                                seg_a, seg_b, 25.0)
        self.assertAlmostEqual(float(got[0]), 25.0, places=9)

    def test_miss_past_the_segment_end(self):
        """A segment fraction outside [0, 1] is past the end of the wall."""
        seg_a = np.array([[1.0, 5.0]])
        seg_b = np.array([[1.0, 6.0]])
        got = ray_segment_range(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]),
                                seg_a, seg_b, 25.0)
        self.assertAlmostEqual(float(got[0]), 25.0, places=9)

    def test_parallel_ray_is_not_a_hit(self):
        """A zero denominator must be masked out, not divided by."""
        seg_a = np.array([[1.0, 0.0]])
        seg_b = np.array([[5.0, 0.0]])
        got = ray_segment_range(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]),
                                seg_a, seg_b, 25.0)
        self.assertTrue(np.isfinite(got[0]))
        self.assertAlmostEqual(float(got[0]), 25.0, places=9)

    def test_nearest_of_several_walls_wins(self):
        seg_a = np.array([[3.0, -1.0], [1.0, -1.0], [7.0, -1.0]])
        seg_b = np.array([[3.0, 1.0], [1.0, 1.0], [7.0, 1.0]])
        got = ray_segment_range(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]),
                                seg_a, seg_b, 25.0)
        self.assertAlmostEqual(float(got[0]), 1.0, places=9)


class TestMapLoading(unittest.TestCase):
    """The occupancy grid and the wall set built from it."""

    def test_grid_matches_the_map_header(self):
        track = _track()
        self.assertAlmostEqual(track.spec.resolution, 0.05, places=9)
        self.assertAlmostEqual(track.spec.origin[0], -8.68, places=9)
        self.assertAlmostEqual(track.spec.origin[1], -1.46, places=9)
        self.assertEqual(track.occupancy_map.ndim, 2)
        self.assertTrue((track.occupancy_map == 0.0).any(), 'no obstacles found')

    def test_walls_span_the_map_extent(self):
        """Catches a missing top-to-bottom flip, which mirrors the whole track.

        Row 0 of an occupancy grid is the smallest world y, but row 0 of the
        image is the top of the picture. If the flip is dropped the wall set is
        still the right size and shape -- it is simply reflected about the map
        centre, and every range is wrong in a way no smoke test would show.
        """
        track = _track()
        walls = scan_sim.wall_segments(track)
        self.assertGreater(len(walls), 100)

        height, width = track.occupancy_map.shape
        res = track.spec.resolution
        ox, oy = track.spec.origin[0], track.spec.origin[1]
        for axis, origin, extent in ((0, ox, width * res), (1, oy, height * res)):
            low = float(min(walls.a[:, axis].min(), walls.b[:, axis].min()))
            high = float(max(walls.a[:, axis].max(), walls.b[:, axis].max()))
            self.assertGreaterEqual(low, origin - res)
            self.assertLessEqual(high, origin + extent + res)

    def test_start_pose_is_not_inside_a_wall(self):
        self.assertFalse(ContactResolver(_track()).detect(_start_cog())[0])


class TestScanGeometry(unittest.TestCase):
    """The sensor frame and the sweep it produces."""

    def setUp(self):
        self.sim = ScanSimulator(_track())

    def test_defaults_match_the_scored_configuration(self):
        """The bridge's sim.yaml, with the referee's noise override."""
        self.assertEqual(self.sim.num_beams, 819)
        self.assertAlmostEqual(self.sim.angle_min, math.radians(-135.0), places=9)
        self.assertAlmostEqual(self.sim.angle_max, math.radians(135.0), places=9)
        self.assertAlmostEqual(self.sim.range_min, 0.05, places=9)
        self.assertAlmostEqual(self.sim.range_max, 25.0, places=9)
        self.assertEqual(self.sim.noise_std, 0.0,
                         'the referee sets lidar_noise_std to 0 for reproducibility')

    def test_sensor_sits_ahead_of_the_rear_axle(self):
        """`_lidar_pose_from_cog`: the tf is from base_link, the pose is CoG."""
        pose = self.sim.lidar_pose((0.0, 0.0, 0.0))
        self.assertAlmostEqual(pose[0], 0.275 - 0.17145, places=9)
        self.assertAlmostEqual(pose[1], 0.0, places=9)

        # Rotated 90 degrees, the same offset must come out along +y.
        pose = self.sim.lidar_pose((0.0, 0.0, math.pi / 2))
        self.assertAlmostEqual(pose[0], 0.0, places=6)
        self.assertAlmostEqual(pose[1], 0.275 - 0.17145, places=6)

    def test_lidar_pose_agrees_with_the_plant_base_link(self):
        """CoG -> base_link -> +0.275 must equal CoG -> sensor."""
        plant = plant_sim.GymPlant()
        plant.reset(START_BASE_LINK)
        bx, by, byaw = plant.base_link_pose
        expected = (bx + 0.275 * math.cos(byaw), by + 0.275 * math.sin(byaw))
        got = self.sim.lidar_pose(_start_cog())
        self.assertAlmostEqual(got[0], expected[0], places=5)
        self.assertAlmostEqual(got[1], expected[1], places=5)

    def test_scan_at_the_spawn_sees_a_corridor(self):
        ranges = self.sim.scan(_start_cog())
        self.assertEqual(ranges.shape, (819,))
        self.assertTrue(np.all(np.isfinite(ranges)))

        def beam(angle):
            return round((angle - self.sim.angle_min) / self.sim.angle_increment)

        left = float(ranges[beam(math.pi / 2)])
        right = float(ranges[beam(-math.pi / 2)])
        ahead = float(ranges[beam(0.0)])
        # The circuit is a narrow corridor; the spawn is on the bottom straight.
        for name, value in (('left', left), ('right', right)):
            self.assertGreater(value, 0.3, f'{name} wall implausibly close')
            self.assertLess(value, 3.0, f'{name} wall implausibly far')
        self.assertGreater(ahead, left, 'the straight should be longer than the width')

    def test_ranges_never_exceed_the_configured_maximum(self):
        ranges = self.sim.scan(_start_cog())
        self.assertLessEqual(float(ranges.max()), self.sim.range_max + 1e-6)

    def test_sub_minimum_ranges_report_zero(self):
        """`_update_scans:495` maps anything under range_min to 0.0, not to the
        raw value -- the driver's nan_to_num never sees it, so a wrong rule here
        shows up as a phantom obstacle at point-blank range."""
        sim = ScanSimulator(_track(), range_min=5.0)
        ranges = sim.scan(_start_cog())
        close = ranges[ranges < 5.0]
        self.assertTrue(np.all(close == 0.0))
        self.assertTrue((ranges == 0.0).any(), 'expected some beams under 5 m')


class TestCullIsExact(unittest.TestCase):
    """The broad phase must be a speed-up, never an approximation."""

    def test_cull_changes_nothing_the_driver_can_see(self):
        cog = _start_cog()
        full = ScanSimulator(_track()).scan(cog)
        culled = ScanSimulator(_track(), cull_reach=9.0).scan(cog)
        clipped_full = np.minimum(full, DRIVER_CLIP_M)
        clipped_culled = np.minimum(culled, DRIVER_CLIP_M)
        np.testing.assert_array_equal(clipped_full, clipped_culled)

    def test_cull_is_exact_below_its_own_radius(self):
        """Below the cull radius the two scans must agree exactly, not closely."""
        cog = _start_cog()
        reach = 9.0
        full = ScanSimulator(_track()).scan(cog)
        culled = ScanSimulator(_track(), cull_reach=reach).scan(cog)
        near = full < reach - 1e-6
        self.assertTrue(near.any(), 'no beams fell inside the cull radius')
        np.testing.assert_array_equal(full[near], culled[near])

    def test_cull_below_range_min_is_rejected(self):
        with self.assertRaises(ValueError):
            ScanSimulator(_track(), range_min=1.0, cull_reach=0.5)


class TestContact(unittest.TestCase):
    """Detection is exact; the response is the documented simplification."""

    def setUp(self):
        self.contact = ContactResolver(_track())

    def test_body_box_is_the_f1tenth_footprint(self):
        self.assertAlmostEqual(self.contact.length, 0.58, places=9)
        self.assertAlmostEqual(self.contact.width, 0.31, places=9)
        # -lr + wheelbase/2: the box is centred between the axles, not on the CoG.
        self.assertAlmostEqual(self.contact.body_dx, -0.17145 + 0.1651, places=9)
        self.assertAlmostEqual(self.contact.body_dy, 0.0, places=9)

    def test_vertices_form_the_right_rectangle(self):
        verts = self.contact.vertices((0.0, 0.0, 0.0))
        self.assertEqual(verts.shape, (4, 2))
        span_x = float(verts[:, 0].max() - verts[:, 0].min())
        span_y = float(verts[:, 1].max() - verts[:, 1].min())
        self.assertAlmostEqual(span_x, 0.58, places=6)
        self.assertAlmostEqual(span_y, 0.31, places=6)

    def _flattest_wall(self):
        """The longest wall segment, with its midpoint and outward normal.

        Deliberately not `walls[0]`: that one is a 7 cm sliver in a map corner,
        where extraction fans a single corner into a dozen segments whose
        normals disagree by 90 degrees. A body placed there is legitimately in
        contact with all of them at once, which makes it useless for testing
        what one wall does. The longest segment is on a straight.
        """
        walls = self.contact.walls
        index = int(np.argmax(walls.length))
        midpoint = 0.5 * (walls.a[index].astype(float) + walls.b[index].astype(float))
        return midpoint, walls.n[index].astype(float), float(walls.length[index])

    def test_body_sitting_on_a_wall_is_a_hit(self):
        """Placed on a wall midpoint, the box must overlap it by construction."""
        midpoint, _, length = self._flattest_wall()
        self.assertGreater(length, self.contact.width,
                           'no wall longer than the car is wide')
        yaw = 0.0
        # `detect` takes a CoG pose, so undo the body offset to land the BOX there.
        cog = (float(midpoint[0]) - self.contact.body_dx * math.cos(yaw),
               float(midpoint[1]) - self.contact.body_dx * math.sin(yaw),
               yaw)
        hit, normal, penetration = self.contact.detect(cog)
        self.assertTrue(hit, 'a body centred on a wall segment was not detected')
        self.assertIsNotNone(normal)
        self.assertAlmostEqual(float(np.hypot(*normal)), 1.0, places=5)
        self.assertGreater(penetration, 0.0)

    def test_response_removes_inward_velocity_without_bouncing(self):
        """Restitution is 0.0 in the judged config, so a hit is a dead stop.

        Driving at a wall must not reverse the normal velocity (that would be a
        bounce), and must not leave it negative (that would be tunnelling).
        """
        midpoint, normal, _ = self._flattest_wall()

        # Nose just over the wall line, travelling straight into it.
        course = math.atan2(-normal[1], -normal[0])
        centre = midpoint + normal * (0.5 * self.contact.length - 0.02)
        state = np.zeros(7)
        state[plant_sim.X] = float(centre[0]) - self.contact.body_dx * math.cos(course)
        state[plant_sim.Y] = float(centre[1]) - self.contact.body_dx * math.sin(course)
        state[plant_sim.PSI] = course
        state[plant_sim.V] = 2.0

        self.assertTrue(self.contact.resolve(state))
        speed = float(state[plant_sim.V])
        heading = float(state[plant_sim.PSI]) + float(state[plant_sim.BETA])
        velocity = np.array([speed * math.cos(heading), speed * math.sin(heading)])
        approach = float(velocity @ normal)
        self.assertGreaterEqual(approach, -1e-6, 'still moving into the wall')
        self.assertLessEqual(approach, 1e-3, 'bounced off a restitution-0 wall')
        self.assertLessEqual(speed, 2.0 + 1e-6, 'contact added energy')

    def test_corner_contact_resolves_every_touching_wall(self):
        """Wedged into a corner, no constraint may be left violated.

        Extraction splits a corner into a fan of short segments with
        conflicting normals. Resolving only the deepest leaves the car moving
        into the others -- which is exactly why this is iterated.
        """
        walls = self.contact.walls
        midpoint = 0.5 * (walls.a[0].astype(float) + walls.b[0].astype(float))
        yaw = 0.0
        cog = (float(midpoint[0]) - self.contact.body_dx * math.cos(yaw),
               float(midpoint[1]) - self.contact.body_dx * math.sin(yaw),
               yaw)
        normals, _ = self.contact.contacts(cog)
        self.assertGreater(len(normals), 1, 'expected a multi-wall corner here')

        for heading in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            state = np.zeros(7)
            state[plant_sim.X], state[plant_sim.Y] = cog[0], cog[1]
            state[plant_sim.PSI] = heading
            state[plant_sim.V] = 2.0
            self.assertTrue(self.contact.resolve(state))

            speed = float(state[plant_sim.V])
            course = float(state[plant_sim.PSI]) + float(state[plant_sim.BETA])
            velocity = np.array([speed * math.cos(course), speed * math.sin(course)])
            worst = float((normals @ velocity).min())
            self.assertGreaterEqual(
                worst, -1e-6,
                f'heading {heading:.2f} still drives into a touching wall')
            self.assertLessEqual(speed, 2.0 + 1e-6, 'contact added energy')

    def test_free_space_is_left_alone(self):
        state = np.zeros(7)
        cog = _start_cog()
        state[plant_sim.X], state[plant_sim.Y], state[plant_sim.PSI] = cog
        state[plant_sim.V] = 2.0
        before = state.copy()
        self.assertFalse(self.contact.resolve(state))
        np.testing.assert_array_equal(state, before)

    def test_response_is_declared_approximate(self):
        """The harness must be able to say the trajectory is not bit-faithful."""
        self.assertTrue(ContactResolver.approximate)


class TestCollisionCounter(unittest.TestCase):
    """`session.register_collision`, reproduced exactly."""

    def test_no_contact_never_counts(self):
        counter = CollisionCounter()
        for step in range(500):
            self.assertFalse(counter.update(step * 0.01, False))
        self.assertEqual(counter.events, 0)

    def test_sustained_contact_counts_once_per_interval(self):
        """Parking on a barrier is not cheaper than hitting it once."""
        counter = CollisionCounter(interval_s=1.0)
        for step in range(300):          # 3.0 s of unbroken contact
            counter.update(step * 0.01, True)
        self.assertEqual(counter.events, 3)
        self.assertEqual(counter.steps_in_contact, 300)

    def test_clock_runs_from_the_last_counted_not_the_last_pulse(self):
        """Pulses inside the window must not push the deadline back."""
        counter = CollisionCounter(interval_s=1.0)
        self.assertTrue(counter.update(0.0, True))
        for step in range(1, 100):       # pulses at 0.01 .. 0.99 s
            self.assertFalse(counter.update(step * 0.01, True))
        self.assertTrue(counter.update(1.0, True))
        self.assertEqual(counter.events, 2)

    def test_penalty_and_disqualification_thresholds(self):
        counter = CollisionCounter(interval_s=1.0, penalty_s=10.0, max_collisions=10)
        for event in range(10):
            counter.update(event * 1.0, True)
        self.assertEqual(counter.events, 10)
        self.assertAlmostEqual(counter.penalty, 100.0, places=9)
        self.assertFalse(counter.disqualified, 'DQ is strictly above the limit')

        counter.update(10.0, True)
        self.assertEqual(counter.events, 11)
        self.assertTrue(counter.disqualified)

    def test_matches_the_scored_baseline_shape(self):
        """7 collisions cost 70 s, as in results/CalcuLOL__20260929T172052Z.json."""
        counter = CollisionCounter()
        for event in range(7):
            counter.update(event * 2.0, True)
        self.assertEqual(counter.events, 7)
        self.assertAlmostEqual(counter.penalty, 70.0, places=9)


if __name__ == '__main__':
    unittest.main()
