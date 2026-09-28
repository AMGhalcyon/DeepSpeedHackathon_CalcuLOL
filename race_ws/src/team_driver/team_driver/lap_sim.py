#!/usr/bin/env python3
"""Closed-loop lap simulator for the MPC driver.

Runs the controller against the VehicleModel along the real icra26 centerline
and reports whether a lap is completed and how close the car came to the walls.

    python3 lap_sim.py                     # defaults
    python3 lap_sim.py --v-max 2.5 --lat-accel 0.6
    python3 lap_sim.py --r-delta-steer 0.01   # show a chattering config fail

The simulator deliberately mirrors driver.py's control path (MPC accel ->
speed command, then clip against the reactive bound) so that what it measures
is what the judge will see.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpc_controller import MPCController
from vehicle_model import VehicleModel

# Half-width of the drivable corridor, m. The f1tenth collision body is
# 0.31 m wide, so the car centre must stay well inside this to be safe.
TRACK_HALF_WIDTH = 0.25

# Oscillation gates. `travelled / mean err / off-track` all read "lap completed"
# for every weight set tried, including the ones chattering between full lock and
# centre, so on their own they cannot tell a stable lap from a violent one.
# These two are what make a config pass or fail on smoothness. Budget is
# ~0.005 rad/step for a smooth 1.5 m/s corner; 0.012 leaves headroom.
MAX_MEAN_SLEW = 0.012      # rad per control step, mean |steer[k] - steer[k-1]|
MAX_LATERAL_ERR = 0.10     # m, max deviation from the centerline


def nearest_error(path, position, hint):
    """Signed lateral offset from the path, searched near `hint`.

    The search window WRAPS. The reference is a closed loop and the sim runs
    past the finish line into lap 2, so a window clipped to `[0, len(path))`
    saturates at the last node the moment the car passes it: `hint` sticks
    there forever and every later reading is the distance to a stale point.
    That is not a metric that reads "badly", it is one that reports the car as
    metres off track while the controller tracks it to within centimetres --
    which is exactly what it did, and why every weight sweep moved the
    "failure point" by less than 1%.
    """
    n = len(path)
    idx = (np.arange(hint - 20, hint + 21)) % n
    seg = path[idx]
    d = np.sum((seg - position) ** 2, axis=1)
    i = int(idx[int(np.argmin(d))])
    nxt = path[(i + 1) % n]
    prv = path[(i - 1) % n]
    tangent = nxt - prv
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    normal = np.array([-tangent[1], tangent[0]])
    return float((position - path[i]) @ normal), i


def run(path, v_max, lat_accel, a_brake, horizon, max_steps, quiet=False,
        weights=None):
    # Weights come from the controller's own defaults unless overridden, so a
    # tuning change cannot land in the YAML and quietly miss what the sim tests.
    mpc = MPCController(
        vehicle_model=VehicleModel(),
        prediction_horizon=horizon,
        control_horizon=horizon,
        dt=0.025,
        **(weights or {}),
    )
    profile = mpc.speed_profile(v_max=v_max, lat_accel=lat_accel,
                                a_brake=a_brake, path=path)
    mpc.set_reference_path(path, profile)

    # Start on the centerline at the first waypoint, aligned with the track.
    start, nxt = path[0], path[1]
    state = np.array([start[0], start[1],
                      math.atan2(nxt[1] - start[1], nxt[0] - start[0]),
                      1.0, 0.0])

    model = VehicleModel()
    lap = float(np.sum(np.linalg.norm(
        np.diff(np.vstack([path, path[:1]]), axis=0), axis=1)))

    travelled = 0.0
    hint = 0
    errors = []
    v_peak = 0.0
    solve_ms = []
    steers = []
    steer = 0.0

    for step in range(max_steps):
        err, hint = nearest_error(path, state[:2], hint)
        errors.append(abs(err))

        t0 = time.perf_counter()
        out, diag = mpc.get_control(state, np.array([steer, 0.0]))
        solve_ms.append((time.perf_counter() - t0) * 1e3)
        steer = float(out[0])
        accel = float(out[1])
        steers.append(steer)
        speed_cmd = max(min(state[3] + accel * mpc.dt, profile.max()), 0.0)

        # The published command is a speed and the sim closes that loop itself,
        # so drive the model with the requested speed, not with the accel.
        applied_accel = (speed_cmd - state[3]) / mpc.dt
        state = model.dynamics_discrete(state, np.array([steer, applied_accel]), mpc.dt)
        travelled += float(state[3]) * mpc.dt
        v_peak = max(v_peak, float(state[3]))

        if not quiet and step % 200 == 0:
            print(f'  step {step:4d}  s={travelled:6.1f}m  '
                  f'err={err:+.3f}m  v={state[3]:.2f}  steer={steer:+.3f}')

    errors = np.array(errors)
    steers = np.array(steers)
    slew = np.abs(np.diff(steers))
    # A reversal is a sign flip in the commanded steering. Zero-deltas are
    # dropped rather than counted: a tie between two identical commands is not
    # a direction change, but counting it as one inflates the total on the
    # settled, near-zero steering a converged controller produces.
    signs = np.sign(np.diff(steers))
    signs = signs[signs != 0]
    reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

    return {
        'lap_len': lap,
        'travelled': travelled,
        'progress': travelled / lap,
        'completed': travelled >= lap,
        'max_err': float(errors.max()),
        'mean_err': float(errors.mean()),
        'off_track': int(np.sum(errors > TRACK_HALF_WIDTH)),
        'v_peak': v_peak,
        'mean_slew': float(slew.mean()) if slew.size else 0.0,
        'max_slew': float(slew.max()) if slew.size else 0.0,
        'steer_reversals': reversals,
        'steer_min': float(steers.min()) if steers.size else 0.0,
        'steer_max': float(steers.max()) if steers.size else 0.0,
        'solve_mean_ms': float(np.mean(solve_ms)),
        'solve_max_ms': float(np.max(solve_ms)),
        'profile_min': float(profile.min()),
        'profile_max': float(profile.max()),
    }


def verdict(r):
    """Overall pass/fail. Smoothness gates are as binding as completing the lap."""
    reasons = []
    if not r['completed']:
        reasons.append('lap not completed')
    if r['off_track']:
        reasons.append(f"{r['off_track']} off-track steps")
    if r['mean_slew'] > MAX_MEAN_SLEW:
        reasons.append(f"mean slew {r['mean_slew']:.4f} > {MAX_MEAN_SLEW}")
    if r['max_err'] > MAX_LATERAL_ERR:
        reasons.append(f"max error {r['max_err']:.3f} > {MAX_LATERAL_ERR} m")
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v-max', type=float, default=3.0)
    ap.add_argument('--lat-accel', type=float, default=0.8)
    ap.add_argument('--a-brake', type=float, default=None)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--max-steps', type=int, default=4000)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--r-delta-steer', type=float, default=None,
                    help='override the steering-rate weight (to test the gate)')
    ap.add_argument('--q-pos', type=float, default=None,
                    help='override the position-tracking weight')
    args = ap.parse_args()

    a_brake = args.a_brake if args.a_brake is not None else args.lat_accel
    here = os.path.dirname(os.path.abspath(__file__))
    path = np.loadtxt(os.path.join(here, '..', '..', '..', '..', 'maps',
                                   'icra26_centerline.csv'),
                      delimiter=',', comments='#')
    path = np.atleast_2d(path)[:, :2]

    overrides = {}
    if args.r_delta_steer is not None:
        overrides['r_delta_steer'] = args.r_delta_steer
    if args.q_pos is not None:
        overrides['q_pos'] = args.q_pos

    r = run(path, args.v_max, args.lat_accel, a_brake, args.horizon,
            args.max_steps, args.quiet, weights=overrides)
    fails = verdict(r)
    if overrides:
        print(f"\nweight overrides: {overrides}")
    print(f"\nv_max={args.v_max} lat_accel={args.lat_accel} horizon={args.horizon}")
    print(f"  lap length      {r['lap_len']:.1f} m")
    print(f"  travelled       {r['travelled']:.1f} m  ({100 * r['progress']:.0f}% of a lap)")
    print(f"  lateral error   mean {r['mean_err']:.3f} m, max {r['max_err']:.3f} m")
    print(f"  off-track steps {r['off_track']} / {args.max_steps}")
    print(f"  steering slew   mean {r['mean_slew']:.4f}, max {r['max_slew']:.4f} rad/step"
          f"  (gate < {MAX_MEAN_SLEW})")
    print(f"  reversals       {r['steer_reversals']}"
          f"  range {r['steer_min']:+.3f} to {r['steer_max']:+.3f} rad")
    print(f"  speed profile   {r['profile_min']:.2f}-{r['profile_max']:.2f} m/s, peak {r['v_peak']:.2f}")
    print(f"  solve time      mean {r['solve_mean_ms']:.1f} ms, max {r['solve_max_ms']:.1f} ms")
    if fails:
        print(f"  VERDICT:        FAIL -- {'; '.join(fails)}")
    else:
        print(f"  VERDICT:        PASS -- lap completed, "
              f"{r['off_track']} off-track, "
              f"slew {r['mean_slew']:.4f} and error {r['max_err']:.3f} m within gates")


if __name__ == '__main__':
    main()
