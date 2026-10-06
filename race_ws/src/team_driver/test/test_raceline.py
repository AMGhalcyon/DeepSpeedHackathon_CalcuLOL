#!/usr/bin/env python3
"""The runtime racing line (Phase 2, Step 2).

Synthetic grids pin the geometry (apex cutting, cone sealing, wall clearance);
the shipped icra26 map pins the acceptance gate: shorter than the centreline,
lower sum of curvature squared, never closer to a sealed wall than the margin,
and a lower lap-time floor.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, 'team_driver'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import lap_floor  # noqa: E402
import raceline as rl  # noqa: E402
import speed_profile as sp  # noqa: E402

RES = 0.05
HALF_WIDTH = 0.155
MARGIN = 0.10
CLEARANCE_TOL = 0.03          # [m] one grid cell of slack on a rasterised wall
FULL_LOCK_KAPPA = 1.269       # [1/m] the tightest turn the steering can follow
SHIPPED_MARGIN = 0.25         # [m] raceline_margin the driver and the cached CSV use
BUILD_HANG_GUARD_SECONDS = 30.0   # measured 3.4 s idle, up to ~15 s on a contended machine
MAP_YAML = os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir,
                        'maps', 'icra26.yaml')


def _rounded_rectangle(width, height, radius, spacing=0.1):
    """Closed centreline of a rounded rectangle, counter-clockwise."""
    pts = []
    corners = [(width - radius, radius, -np.pi / 2), (width - radius, height - radius, 0.0),
               (radius, height - radius, np.pi / 2), (radius, radius, np.pi)]
    for cx, cy, start in corners:
        a = np.linspace(start, start + np.pi / 2, max(8, int(np.pi / 2 * radius / spacing)),
                        endpoint=False)
        pts.append(np.column_stack([cx + radius * np.cos(a), cy + radius * np.sin(a)]))
    return rl.resample_closed(np.vstack(pts), spacing)


def _track_grid(centerline, half_width, pad=1.0):
    """A grid whose only free space is a corridor of `half_width` round `centerline`."""
    lo = centerline.min(axis=0) - half_width - pad
    hi = centerline.max(axis=0) + half_width + pad
    shape = np.ceil((hi - lo)[::-1] / RES).astype(int)
    seeds = np.zeros(shape, dtype=bool)
    cols = ((centerline[:, 0] - lo[0]) / RES).astype(int)
    rows = (shape[0] - 1 - ((centerline[:, 1] - lo[1]) / RES)).astype(int)
    seeds[rows, cols] = True
    free = ndimage.distance_transform_edt(~seeds) * RES <= half_width
    return rl.OccupancyGrid(occupied=~free, resolution=RES, origin=(float(lo[0]), float(lo[1])))


def _sum_kappa_sq(path):
    path = rl.resample_closed(path, 0.1)
    kappa = sp.closed_curvature(path)
    return float(np.sum(kappa ** 2) * 0.1)


def _length(path):
    return float(np.sum(sp.segment_lengths(path, True)))


class TestGridAndSealing(unittest.TestCase):

    def test_world_cell_round_trip(self):
        grid = rl.OccupancyGrid(occupied=np.zeros((40, 60), bool), resolution=0.05,
                                origin=(-1.0, 2.0))
        xy = np.array([[0.123, 2.456], [1.5, 3.4]])
        back = grid.cell_to_world(*grid.world_to_cell(xy))
        np.testing.assert_allclose(back, xy, atol=RES)

    def _cone_row(self, gap):
        occ = np.zeros((120, 120), bool)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        cone = int(round(0.3 / RES))
        step = cone + int(round(gap / RES))
        for c in range(20, 100, step):
            occ[55:55 + cone, c:c + cone] = True
        return rl.OccupancyGrid(occupied=occ, resolution=RES, origin=(0.0, 0.0))

    def test_cone_gap_of_0p4_m_is_sealed(self):
        grid = self._cone_row(gap=0.4)
        sealed = rl.seal_cones(grid, radius=0.3)
        row = 55 + 2
        self.assertTrue(sealed.occupied[row, 20:90].all(),
                        'a 0.4 m gap between cones is still open')

    def test_one_metre_corridor_stays_open(self):
        occ = np.zeros((120, 120), bool)
        occ[:50, :] = True
        occ[70:, :] = True                       # a 20-cell = 1.0 m corridor
        grid = rl.OccupancyGrid(occupied=occ, resolution=RES, origin=(0.0, 0.0))
        sealed = rl.seal_cones(grid, radius=0.3)
        self.assertFalse(sealed.occupied[55:65, 10:110].any(),
                         'sealing closed a corridor wide enough to drive')


class TestSyntheticTrack(unittest.TestCase):

    def setUp(self):
        self.center = _rounded_rectangle(8.0, 5.0, 1.2)
        self.grid = _track_grid(self.center, half_width=0.8)
        self.line = rl.build_raceline(self.center, self.grid, margin=MARGIN,
                                      half_width=HALF_WIDTH)

    def test_line_is_shorter_with_lower_curvature_than_the_centreline(self):
        self.assertLess(_length(self.line.xy), _length(self.center))
        self.assertLess(_sum_kappa_sq(self.line.xy), _sum_kappa_sq(self.center))

    def test_line_keeps_the_required_clearance_from_the_walls(self):
        sealed = rl.seal_cones(self.grid, radius=0.3)
        clearance = rl.wall_clearance(sealed, self.line.xy)
        self.assertGreaterEqual(float(clearance.min()),
                                HALF_WIDTH + MARGIN - CLEARANCE_TOL)

    def test_line_cuts_the_apex_of_a_corner(self):
        # Centre of the first arc (counter-clockwise from the bottom-right).
        corner = np.array([8.0 - 1.2, 1.2])
        quadrant = lambda p: (p[:, 0] > corner[0]) & (p[:, 1] < corner[1])  # noqa: E731
        d_line = np.linalg.norm(self.line.xy[quadrant(self.line.xy)] - corner, axis=1)
        d_center = np.linalg.norm(self.center[quadrant(self.center)] - corner, axis=1)
        self.assertLess(float(d_line.min()), float(d_center.min()) - 0.1,
                        'the line never leaves the centreline radius at the apex')

    def test_table_columns_are_consistent(self):
        t = rl.raceline_table(self.line.xy, a_lat_max=5.0, lock_margin=0.95,
                              a_accel_max=9.51, a_brake_max=9.51, v_max=20.0)
        self.assertEqual(t.shape[1], 7)
        s, x, y, psi, kappa, vx, ax = t.T
        self.assertTrue(np.all(np.diff(s) > 0))
        np.testing.assert_allclose(np.hypot(np.diff(x), np.diff(y)), np.diff(s), rtol=0.02)
        self.assertTrue(np.all(vx > 0))
        # ax is the acceleration implied by vx along s.
        ds = np.linalg.norm(np.roll(t[:, 1:3], -1, axis=0) - t[:, 1:3], axis=1)
        np.testing.assert_allclose(ax, (np.roll(vx, -1) ** 2 - vx ** 2) / (2 * ds), atol=1e-6)


class TestMinTimeRefinement(unittest.TestCase):
    """`refine_min_time` on a synthetic track: faster, and never closer to a wall."""

    PROFILE = dict(a_lat_max=5.0, lock_margin=0.95, a_accel_max=9.51, a_brake_max=6.0,
                   v_max=10.0, sector_scale=())

    @classmethod
    def setUpClass(cls):
        center = _rounded_rectangle(8.0, 5.0, 1.2)
        grid = _track_grid(center, half_width=0.8)
        cls.base = rl.build_raceline(center, grid, margin=MARGIN, half_width=HALF_WIDTH)
        cls.required = HALF_WIDTH + MARGIN
        cls.refined = rl.refine_min_time(cls.base.xy, cls.base.sealed, cls.required,
                                         profile=cls.PROFILE, control_points=12, rounds=1)

    def _lap(self, xy):
        return sp.ideal_lap_time(xy, sp.speed_profile(xy, **self.PROFILE))

    def test_refined_line_is_faster_at_the_profile_it_was_refined_for(self):
        self.assertLess(self._lap(self.refined), self._lap(self.base.xy))

    def test_refined_line_keeps_the_required_clearance(self):
        clearance = rl.wall_clearance(self.base.sealed, self.refined)
        self.assertGreaterEqual(float(clearance.min()), self.required - rl.CLEARANCE_EPS)

    def test_refined_line_is_delivered_at_the_output_spacing(self):
        np.testing.assert_allclose(sp.segment_lengths(self.refined, True),
                                   rl.OUTPUT_SPACING, rtol=0.05)

    def test_a_refinement_that_is_not_faster_is_refused(self):
        # Zero rounds leaves the line as it came: nothing gained, nothing to ship.
        with self.assertRaises(rl.RacelineError):
            rl.refine_min_time(self.base.xy, self.base.sealed, self.required,
                               profile=self.PROFILE, rounds=0)

    def test_a_line_that_cannot_keep_the_clearance_is_refused(self):
        with self.assertRaises(rl.RacelineError):
            rl.refine_min_time(self.base.xy, self.base.sealed, 0.9, profile=self.PROFILE,
                               control_points=12, rounds=1, max_iter=2)


class TestWeightedSolve(unittest.TestCase):
    """Per-row weights on the second differences of `_min_curvature_offsets`."""

    def setUp(self):
        center = _rounded_rectangle(8.0, 5.0, 1.2)
        sealed = rl.seal_cones(_track_grid(center, half_width=0.8), 0.3)
        self.ref = rl.resample_closed(center, rl.OPT_SPACING)
        self.normals = rl._left_normals(self.ref)
        self.lower, self.upper = rl._offset_bounds(
            sealed, rl._clearance_map(sealed), self.ref, self.normals, HALF_WIDTH + MARGIN)

    def _solve(self, **kwargs):
        return rl._min_curvature_offsets(self.ref, self.normals, self.lower, self.upper,
                                         **kwargs)

    def test_unit_weights_reproduce_the_uniform_solve(self):
        np.testing.assert_allclose(self._solve(weights=np.ones(len(self.ref))), self._solve(),
                                   atol=1e-9)

    def test_uneven_weights_change_the_solution(self):
        weights = np.where(self.ref[:, 0] > 6.0, 8.0, 1.0)
        shift = np.abs(self._solve(weights=weights) - self._solve())
        self.assertGreater(float(shift.max()), 1e-3)

    def test_weights_must_match_the_rows_and_be_positive(self):
        n = len(self.ref)
        for name, bad in (('short', np.ones(n - 1)), ('zero', np.zeros(n)),
                          ('negative', -np.ones(n)), ('nan', np.full(n, np.nan))):
            with self.subTest(weights=name), self.assertRaises(ValueError):
                self._solve(weights=bad)


class TestReweightedBuild(unittest.TestCase):
    """The build reweights by curvature, and falls back to the uniform solve."""

    def setUp(self):
        self.center = _rounded_rectangle(8.0, 5.0, 1.2)
        self.grid = _track_grid(self.center, half_width=0.8)

    def _build(self, **kwargs):
        return rl.build_raceline(self.center, self.grid, margin=MARGIN,
                                 half_width=HALF_WIDTH, **kwargs)

    def test_the_default_build_is_reweighted(self):
        self.assertTrue(self._build().reweighted)

    def test_reweight_off_gives_the_uniform_line(self):
        self.assertFalse(self._build(reweight=False).reweighted)

    def test_non_convergence_falls_back_to_the_uniform_solve(self):
        original = rl._min_curvature_offsets

        def fails_when_weighted(*args, weights=None, **kwargs):
            if weights is not None:
                raise rl.RacelineError('raceline QP did not converge: boom')
            return original(*args, **kwargs)

        with mock.patch.object(rl, '_min_curvature_offsets', fails_when_weighted), \
                self.assertWarns(RuntimeWarning):
            line = self._build()
        self.assertFalse(line.reweighted)
        np.testing.assert_allclose(line.xy, self._build(reweight=False).xy, atol=1e-9)
        clearance = rl.wall_clearance(line.sealed, line.xy)
        self.assertGreaterEqual(float(clearance.min()), HALF_WIDTH + MARGIN - CLEARANCE_TOL)

    def test_clearance_shortfall_also_falls_back_to_the_uniform_solve(self):
        original = rl._relinearise

        def weighted_line_hugs_the_walls(centerline, *args, kappa_ref, **kwargs):
            if kappa_ref is not None:
                return rl.resample_closed(centerline + 0.7, rl.OPT_SPACING)
            return original(centerline, *args, kappa_ref=kappa_ref, **kwargs)

        with mock.patch.object(rl, '_relinearise', weighted_line_hugs_the_walls), \
                self.assertWarns(RuntimeWarning) as caught:
            line = self._build()
        self.assertIn('closer than', str(caught.warning))
        self.assertFalse(line.reweighted)
        clearance = rl.wall_clearance(line.sealed, line.xy)
        self.assertGreaterEqual(float(clearance.min()), HALF_WIDTH + MARGIN - CLEARANCE_TOL)

    def test_a_failing_uniform_solve_still_raises(self):
        with mock.patch.object(rl, '_min_curvature_offsets',
                               side_effect=rl.RacelineError('raceline QP did not converge')), \
                self.assertWarns(RuntimeWarning), self.assertRaises(rl.RacelineError):
            self._build()

    def test_an_unrelated_runtime_error_is_not_swallowed(self):
        with mock.patch.object(rl, '_min_curvature_offsets',
                               side_effect=RuntimeError('something else broke')):
            with self.assertRaisesRegex(RuntimeError, 'something else broke'):
                self._build()

    def test_nonsense_reweighting_parameters_are_refused(self):
        for name, kwargs in (('zero kappa_ref', dict(kappa_ref=0.0)),
                             ('negative kappa_ref', dict(kappa_ref=-0.6)),
                             ('no iterations', dict(reweight_iterations=0))):
            with self.subTest(case=name), self.assertRaises(ValueError):
                self._build(**kwargs)


@unittest.skipUnless(os.path.exists(MAP_YAML), 'maps/icra26.yaml not found')
class TestIcra26Reweighted(unittest.TestCase):
    """The shipped line (margin 0.25) must be steerable through the sector-D apex."""

    @classmethod
    def setUpClass(cls):
        cls.grid = rl.load_occupancy(MAP_YAML)
        cls.center = lap_floor.load_path(lap_floor.default_centerline_path())
        with mock.patch.object(rl, '_min_curvature_offsets',
                               wraps=rl._min_curvature_offsets) as solves:
            cls.line = rl.build_raceline(cls.center, cls.grid, margin=SHIPPED_MARGIN,
                                         half_width=HALF_WIDTH)
        cls.solve_count = solves.call_count
        cls.uniform = rl.build_raceline(cls.center, cls.grid, margin=SHIPPED_MARGIN,
                                        half_width=HALF_WIDTH, reweight=False)

    def test_the_build_did_not_silently_fall_back(self):
        self.assertTrue(self.line.reweighted)

    def test_peak_smoothed_curvature_is_below_full_lock(self):
        peak = float(sp.closed_curvature(self.line.xy).max())
        self.assertLess(peak, FULL_LOCK_KAPPA)
        self.assertLess(peak, float(sp.closed_curvature(self.uniform.xy).max()) - 0.05)

    def test_clearance_is_at_least_the_required_distance(self):
        clearance = rl.wall_clearance(self.line.sealed, self.line.xy)
        self.assertGreaterEqual(float(clearance.min()),
                                HALF_WIDTH + SHIPPED_MARGIN - rl.CLEARANCE_EPS)

    def test_build_cost_is_one_solve_per_iteration_with_no_slack_retry(self):
        # Wall time swings 2-5x with BLAS threads and machine load, so the cost is
        # pinned by the work done; the clock only guards against a hang.
        self.assertEqual(self.solve_count, rl.REWEIGHT_ITERATIONS)
        self.assertLess(self.line.build_seconds, BUILD_HANG_GUARD_SECONDS)


@unittest.skipUnless(os.path.exists(MAP_YAML), 'maps/icra26.yaml not found')
class TestIcra26MinTime(unittest.TestCase):
    """The shipped line: the reweighted QP line, refined for lap time (~60 s)."""

    CACHE = os.path.join(_HERE, os.pardir, 'config', 'icra26_raceline.csv')
    REQUIRED_GAIN = 0.3            # [s] of ideal lap at the profile it was refined for

    @classmethod
    def setUpClass(cls):
        grid = rl.load_occupancy(MAP_YAML)
        center = lap_floor.load_path(lap_floor.default_centerline_path())
        cls.qp = rl.build_raceline(center, grid, margin=SHIPPED_MARGIN, half_width=HALF_WIDTH)
        cls.refined = rl.refine_min_time(cls.qp.xy, cls.qp.sealed, HALF_WIDTH + SHIPPED_MARGIN)

    @staticmethod
    def _lap(xy):
        return sp.ideal_lap_time(xy, sp.speed_profile(xy, **rl.REFINE_PROFILE))

    def test_ideal_lap_improves_by_at_least_0p3_s(self):
        self.assertLess(self._lap(self.refined), self._lap(self.qp.xy) - self.REQUIRED_GAIN)

    def test_clearance_is_at_least_the_required_distance(self):
        clearance = rl.wall_clearance(self.qp.sealed, self.refined)
        self.assertGreaterEqual(float(clearance.min()),
                                HALF_WIDTH + SHIPPED_MARGIN - rl.CLEARANCE_EPS)

    def test_peak_smoothed_curvature_is_below_full_lock(self):
        self.assertLess(float(sp.closed_curvature(self.refined).max()), FULL_LOCK_KAPPA)

    def test_shipped_csv_is_a_reproducible_min_time_line(self):
        # Not point for point: BLAS threading moves the QP line by ~1e-6 m and the
        # finite-difference search amplifies that to ~9 cm and ~0.04 s of ideal lap
        # (OMP_NUM_THREADS=1 reproduces the CSV to its 4 decimals). The driver drives
        # the CSV itself, so what must reproduce is the line's quality: method,
        # margin, clearance, the 0.3 s gain, and an ideal lap a fresh run agrees with.
        xy, margin = rl.read_cached(self.CACHE)
        self.assertEqual(margin, SHIPPED_MARGIN)
        self.assertEqual(rl.cached_method(self.CACHE), rl.MIN_TIME_METHOD)
        self.assertGreaterEqual(float(rl.wall_clearance(self.qp.sealed, xy).min()),
                                HALF_WIDTH + SHIPPED_MARGIN - rl.CLEARANCE_EPS)
        self.assertLess(self._lap(xy), self._lap(self.qp.xy) - self.REQUIRED_GAIN)
        self.assertAlmostEqual(self._lap(xy), self._lap(self.refined), delta=0.05,
                               msg='config/icra26_raceline.csv is stale: rerun '
                                   'OMP_NUM_THREADS=1 raceline.py --refine --write')


@unittest.skipUnless(os.path.exists(MAP_YAML), 'maps/icra26.yaml not found')
class TestIcra26(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.grid = rl.load_occupancy(MAP_YAML)
        cls.center = lap_floor.load_path(lap_floor.default_centerline_path())
        cls.line = rl.build_raceline(cls.center, cls.grid, margin=MARGIN,
                                     half_width=HALF_WIDTH)

    def test_raceline_is_shorter_and_smoother_than_the_centreline(self):
        self.assertLess(_length(self.line.xy), _length(self.center))
        self.assertLess(_sum_kappa_sq(self.line.xy), _sum_kappa_sq(self.center))

    def test_no_point_is_within_the_margin_of_a_sealed_wall(self):
        sealed = rl.seal_cones(self.grid, radius=0.3)
        clearance = rl.wall_clearance(sealed, self.line.xy)
        self.assertGreaterEqual(float(clearance.min()),
                                HALF_WIDTH + MARGIN - CLEARANCE_TOL,
                                f'closest approach {float(clearance.min()):.3f} m')

    def test_every_cone_link_is_fully_sealed(self):
        sealed = rl.seal_cones(self.grid, radius=0.3)
        links = rl.cone_links(self.grid)
        self.assertGreaterEqual(len(links), 15, 'the icra26 cone rows were not found')
        t = np.linspace(0.0, 1.0, 60)[:, None]
        for a, b in links:
            rows, cols = sealed.world_to_cell(a + t * (b - a))
            self.assertTrue(sealed.occupied[rows, cols].all(), f'gap left between {a} and {b}')

    def test_line_crosses_no_cone_link_and_keeps_clear_of_the_sealed_rows(self):
        links = rl.cone_links(self.grid)
        line = self.line.xy
        p, q = line, np.roll(line, -1, axis=0)
        for a, b in links:
            def side(u, v, w):
                return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) \
                    - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
            crosses = ((side(a, b, p) * side(a, b, q)) < 0) & ((side(p, q, a) * side(p, q, b)) < 0)
            self.assertFalse(crosses.any(), f'the line crosses the cone link {a}-{b}')

    def test_floor_is_lower_on_the_raceline(self):
        kw = dict(lock_margin=0.95)
        on_center = lap_floor.floor_table(self.center, (10.0,), **kw)[0].lap_time
        on_line = lap_floor.floor_table(self.line.xy, (10.0,), **kw)[0].lap_time
        self.assertLess(on_line, on_center)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(('info', m))

    def warn(self, m):
        self.lines.append(('warn', m))

    def error(self, m):
        self.lines.append(('error', m))

    def text(self, level):
        return ' '.join(m for lv, m in self.lines if lv == level)


@unittest.skipUnless(os.path.exists(MAP_YAML), 'maps/icra26.yaml not found')
class TestDriverCoreReference(unittest.TestCase):
    """`reference: raceline|centerline`, with the cached CSV as the fallback."""

    @staticmethod
    def _core(**cfg):
        import driver_core
        log = _Log()
        core = driver_core.DriverCore(driver_core.DriverConfig(**cfg), logger=log)
        return core, log

    CACHE = os.path.join(_HERE, os.pardir, 'config', 'icra26_raceline.csv')

    def setUp(self):
        import driver_core
        driver_core._built_raceline.cache_clear()
        driver_core._offline_min_time_line.cache_clear()
        self.addCleanup(driver_core._offline_min_time_line.cache_clear)
        self.addCleanup(driver_core._built_raceline.cache_clear)

    def test_default_reference_is_the_raceline(self):
        core, _ = self._core()
        self.assertNotEqual(len(core.reference_path), 392)
        self.assertLess(_length(core.reference_path), 75.0)

    def test_the_offline_min_time_line_is_driven_first(self):
        core, log = self._core()
        xy, _ = rl.read_cached(self.CACHE)
        np.testing.assert_array_equal(core.reference_path, xy)
        self.assertIn('min-time', log.text('info'))

    def test_a_csv_that_is_not_min_time_is_left_to_the_runtime_build(self):
        with mock.patch.object(rl, 'cached_method', return_value=rl.MIN_CURVATURE_METHOD):
            core, log = self._core()
        self.assertIn('built in', log.text('info'))
        self.assertIn('min-curvature', log.text('warn'))

    def test_a_min_time_line_closer_to_a_wall_than_the_margin_is_not_driven(self):
        with mock.patch.object(rl, 'wall_clearance', return_value=np.zeros(3)):
            core, log = self._core()
        self.assertIn('built in', log.text('info'))
        self.assertIn('clearance', log.text('warn'))

    def test_a_min_time_line_that_failed_its_map_check_is_not_the_last_resort(self):
        # The build failing too must not send the car back to the line it just refused.
        boom = RuntimeError('boom')
        with mock.patch.object(rl, 'wall_clearance', return_value=np.zeros(3)), \
                mock.patch.object(rl, 'build_raceline', side_effect=boom):
            core, log = self._core()
        self.assertEqual(len(core.reference_path), 392)
        self.assertIn('map check', log.text('error'))

    def test_the_centreline_is_still_available(self):
        core, _ = self._core(reference='centerline')
        self.assertEqual(len(core.reference_path), 392)

    def test_raceline_reference_is_shorter_than_the_centreline(self):
        core, log = self._core(reference='raceline')
        self.assertLess(_length(core.reference_path), 75.0)
        self.assertIn('raceline', log.text('info').lower())
        self.assertEqual(len(core.mpc_controller.reference_speeds), len(core.reference_path))

    def test_sector_scale_slows_the_profile_only_inside_its_sector(self):
        import speed_profile as sp
        base, _ = self._core(a_lat_max=8.0, sector_scale=(0.0, 1.0, 1.0))  # sectors off
        backed_off, _ = self._core(a_lat_max=8.0, sector_scale=(30.0, 36.0, 0.5))
        path = base.reference_path
        s = np.concatenate([[0.0], np.cumsum(sp.segment_lengths(path, True)[:-1])])
        corner = (s >= 32.0) & (s < 35.0)
        far = (s >= 55.0) & (s < 70.0)
        self.assertTrue(np.all(backed_off.reference_speeds[corner]
                               < base.reference_speeds[corner] - 0.3))
        np.testing.assert_allclose(backed_off.reference_speeds[far],
                                   base.reference_speeds[far])

    def test_on_the_centreline_the_sectors_become_their_lowest_uniform_budget(self):
        # Sectors are tuned to the raceline's arc axis; on the centreline they would
        # land on the wrong corners, so the whole lap gets the most cautious budget.
        import speed_profile as sp
        core, log = self._core(reference='centerline', a_lat_max=10.0,
                               sector_scale=(30.0, 36.0, 0.8, 38.0, 43.0, 0.5))
        uniform = sp.speed_profile(core.reference_path, a_lat_max=5.0, lock_margin=0.95,
                                   a_accel_max=9.51, a_brake_max=6.0, v_max=10.0)
        np.testing.assert_allclose(core.reference_speeds, uniform)
        self.assertIn('sector', log.text('warn'))

    def test_a_sector_that_lowers_no_waypoint_is_warned_about(self):
        _, log = self._core(a_lat_max=8.0, sector_scale=(500.0, 510.0, 0.5))
        self.assertIn('500', log.text('warn'))

    def _not_min_time(self):
        """The CSV read as a plain min-curvature line, so the runtime build goes first."""
        patcher = mock.patch.object(rl, 'cached_method', return_value=rl.MIN_CURVATURE_METHOD)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_failed_build_falls_back_to_the_cached_csv(self):
        import driver_core
        self._not_min_time()
        original = rl.build_raceline
        rl.build_raceline = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom'))
        driver_core._built_raceline.cache_clear()       # or the memo hides the failure
        try:
            core, log = self._core(reference='raceline')
        finally:
            rl.build_raceline = original
        self.assertIn('cached', log.text('warn').lower() + log.text('error').lower())
        self.assertLess(_length(core.reference_path), 75.0)

    def test_a_uniform_fallback_build_is_not_driven_with_the_tuned_profile(self):
        # The sectors and a_lat_max are tuned to the reweighted line. If the weighted
        # solve fails and build_raceline returns the uniform line instead, the driver
        # must take the cached reweighted line, not run the tuned profile on that one.
        import driver_core
        self._not_min_time()
        cached_xy, _ = rl.read_cached(
            os.path.join(_HERE, os.pardir, 'config', 'icra26_raceline.csv'))
        uniform_stand_in = rl.Raceline(xy=cached_xy * 2.0, sealed=None,
                                       build_seconds=0.0, reweighted=False)
        original = rl.build_raceline
        rl.build_raceline = lambda *a, **k: uniform_stand_in
        driver_core._built_raceline.cache_clear()       # or the memo hides the failure
        try:
            core, log = self._core(reference='raceline')
        finally:
            rl.build_raceline = original
            driver_core._built_raceline.cache_clear()   # do not leak the stand-in
        self.assertLess(_length(core.reference_path), 75.0,
                        'drove the uniform line the tuned profile does not fit')
        self.assertIn('uniform', log.text('warn').lower() + log.text('error').lower())

    def test_unknown_reference_is_refused(self):
        import driver_params
        values = dict(driver_params.DEFAULTS)
        values['reference'] = 'racing_line'
        with self.assertRaises(ValueError):
            driver_params.validate(values)


@unittest.skipUnless(os.path.exists(MAP_YAML), 'maps/icra26.yaml not found')
class TestCachedRaceline(unittest.TestCase):

    CACHE = os.path.join(_HERE, os.pardir, 'config', 'icra26_raceline.csv')

    def test_unusable_caches_are_refused(self):
        import tempfile
        cases = {'few rows': '# margin_m: 0.1\n1,2\n3,4\n',
                 'no margin header': ''.join(f'{i},0\n' for i in range(60)),
                 'open lap': '# margin_m: 0.1\n' + ''.join(f'{i},0\n' for i in range(60)),
                 'nan': '# margin_m: 0.1\n' + ''.join(f'{i},nan\n' for i in range(60))}
        for name, text in cases.items():
            with self.subTest(case=name), tempfile.NamedTemporaryFile(
                    'w', suffix='.csv', delete=False) as handle:
                handle.write(text)
            try:
                with self.assertRaises(ValueError):
                    rl.read_cached(handle.name)
            finally:
                os.unlink(handle.name)

    def test_cache_with_less_margin_than_requested_is_not_used(self):
        import driver_core
        log = _Log()
        original = rl.build_raceline
        rl.build_raceline = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom'))
        driver_core._built_raceline.cache_clear()       # or the memo hides the failure
        try:
            core = driver_core.DriverCore(
                driver_core.DriverConfig(reference='raceline', raceline_margin=0.30), logger=log)
        finally:
            rl.build_raceline = original
        self.assertEqual(len(core.reference_path), 392, 'fell back to a thinner-margin cache')
        self.assertIn('margin', log.text('error'))

    def test_write_refuses_a_line_that_fell_back_to_the_uniform_solve(self):
        import tempfile
        xy, _ = rl.read_cached(self.CACHE)
        grid = rl.load_occupancy(MAP_YAML)
        uniform = rl.Raceline(xy=xy, sealed=grid, build_seconds=0.0, reweighted=False)
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, 'line.csv')
            with mock.patch.object(rl, 'build_raceline', return_value=uniform):
                status = rl.main(['--margin', '0.25', '--write', target])
            self.assertNotEqual(status, 0)
            self.assertFalse(os.path.exists(target), 'a uniform line was cached')

    def test_write_records_how_the_line_was_made(self):
        import tempfile
        xy, _ = rl.read_cached(self.CACHE)
        line = rl.Raceline(xy=xy, sealed=rl.load_occupancy(MAP_YAML), build_seconds=0.0,
                           reweighted=True)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(rl, 'build_raceline', return_value=line), \
                mock.patch.object(rl, 'refine_min_time', side_effect=lambda xy, *a, **k: xy):
            cases = (([], rl.MIN_CURVATURE_METHOD), (['--refine'], rl.MIN_TIME_METHOD))
            for flags, method in cases:
                target = os.path.join(folder, f'{method}.csv')
                with self.subTest(method=method):
                    self.assertEqual(rl.main(['--margin', '0.25', '--write', target] + flags), 0)
                    self.assertEqual(rl.cached_method(target), method)
                    self.assertEqual(rl.read_cached(target)[1], 0.25)

    def test_a_csv_without_a_method_header_is_min_curvature(self):
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as handle:
            handle.write('# margin_m: 0.25\n' + ''.join(f'{i},0\n' for i in range(60)))
        self.addCleanup(os.unlink, handle.name)
        self.assertEqual(rl.cached_method(handle.name), rl.MIN_CURVATURE_METHOD)

    def test_negative_or_nan_margin_is_refused(self):
        import driver_params
        for bad in (-0.01, float('nan')):
            with self.subTest(margin=bad), self.assertRaises(ValueError):
                driver_params.validate({**driver_params.DEFAULTS, 'raceline_margin': bad})


if __name__ == '__main__':
    unittest.main()
