#!/usr/bin/env python3
"""Closed-loop harness: the judged plant, sensors and timing, driving DriverCore.

    python3 lap_sim.py                          # 2 laps, judged timing
    python3 lap_sim.py --laps 10 --json r.json  # a scored-length run
    python3 lap_sim.py --stale-steps 0          # what Step 3's QoS fix is worth
    python3 lap_sim.py --plant instant          # ablation: no steering servo

WHY THE TIMING MODEL
    The judged run sets `use_sim_time_bridge`, which zeroes both bridge timers:
    the bridge steps physics and publishes scan+odom as fast as the CPU allows
    (RTF ~1.4). The driver solves one scan in ~30-40 ms of wall time, so its
    depth-10 subscription queues are always full and every scan it takes is ~10
    physics steps old. While it solves, the sim keeps running -- so each command
    is also held for `solve_wall * rtf` of sim time. Together that is ~150 ms of
    dead time, against an MPC that plans 200 ms ahead and a steering servo that
    needs 290 ms for a full reversal. A harness that applies commands instantly
    to the current state -- the old `lap_sim.py` -- cannot see any of it.

WHAT IS EXACT, WHAT IS NOT
    Plant, servo, LiDAR, collision detection and the referee's 1 Hz collision
    debounce are the judge's own code or line-for-line ports (plant_sim,
    scan_sim). Collision RESPONSE is a restitution-0 approximation, and the
    queue model is a fixed staleness rather than a simulated DDS executor.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plant_sim  # noqa: E402
import scan_sim  # noqa: E402
from driver_core import DriverConfig, DriverCore  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

PHYSICS_DT = 0.01               # s, gym_bridge.py sim_timestep
# Scan and odom are both 9 physics steps old when the driver reads them: the
# depth-10 queue stays full, and its oldest entry is 9 behind the newest.
# Measured live with a /clock probe (run 20260930T091032Z, mode 9 of 1536).
DEFAULT_STALE_STEPS = 9
DEFAULT_RTF = 1.4               # sim seconds per wall second, measured live
# Live, consecutive scans the driver used were ~74 ms of sim apart (the hold),
# against ~32 ms for core.step here: 74 / (32 * 1.4) = 1.65.
DEFAULT_SOLVE_SCALE = 1.65
SCAN_CULL_REACH = 9.0           # m; exact below it, and the driver clips at 8 m

# maps/tracks.yaml: spawn behind the line, and the line itself.
SPAWN_POSE = (-1.68, -0.01, 0.0)
FINISH_LINE_X = 1.32
FINISH_LINE_Y = (-0.76, 0.89)

# The node's declared parameters (driver.py). DriverCore's own fallback builds
# horizon 10, and driver_params.yaml never loads, so these are what the judge runs.
JUDGED_MPC = dict(prediction_horizon=8, control_horizon=3, dt=0.025,
                  q_pos=2.0, q_yaw=2.0, q_vel=0.5, r_steer=0.08, r_accel=0.01,
                  r_delta_steer=0.8, r_delta_accel=0.005,
                  block_map=[0, 1, 1, 2, 2, 2, 2, 2], maxiter=25,
                  solve_budget_ms=15.0)

# Controller gates. The slew gate is the old harness's, kept so the two can be
# compared; the saturation gate is what Step 6 has to reach.
MAX_MEAN_CMD_SLEW = 0.012       # rad per control step
MAX_SAT_DUTY = 0.05             # fraction of physics steps rate-saturated
# Step 6: the wheel estimate handed to the MPC, against the plant's own wheel
# at the same instant. Past this the actuator state does net harm.
MAX_DELTA_EST_ERR = 0.02        # rad
# Step 2's binding gate on the harness itself: today's controller must fail the
# way it fails when judged (7 collisions / 10 laps in the judged run).
GATE_MIN_COLLISIONS_PER_10 = 3
# Rate-saturation duty of the live command stream (run 20260930T091032Z replayed
# through the judge's servo). A faithful harness lands either side of it, so the
# gate asks for agreement with it rather than a one-sided threshold.
LIVE_SAT_DUTY = 0.207
GATE_SAT_TOLERANCE = 0.03
JUDGED_BASELINE = 'results/CalcuLOL__20260929T172052Z.json: 10 laps, 7 collisions'


@dataclass(frozen=True)
class HarnessConfig:
    laps: int = 2
    max_time: Optional[float] = None  # sim s; default scales with laps
    stale_steps: int = DEFAULT_STALE_STEPS
    rtf: float = DEFAULT_RTF
    fixed_solve_ms: Optional[float] = None
    solve_scale: float = DEFAULT_SOLVE_SCALE
    plant: str = 'st'
    seed: Optional[int] = None  # plant noise RNG; inert unless noise is on
    quiet: bool = False

    @property
    def time_limit(self):
        return self.max_time if self.max_time is not None else 30.0 + 90.0 * self.laps


def hold_steps(solve_wall_s, rtf, dt=PHYSICS_DT):
    """Physics steps the sim advances while the driver solves one command."""
    return max(1, int(round(solve_wall_s * rtf / dt)))


def hold_for(cfg, measured_solve_s):
    """Hold for one command: the fixed solve time, else the scaled measured one."""
    if cfg.fixed_solve_ms is not None:
        return hold_steps(cfg.fixed_solve_ms / 1e3, cfg.rtf)
    return hold_steps(measured_solve_s * cfg.solve_scale, cfg.rtf)


class SnapshotRing:
    """The last few published (state, stamp) pairs; the driver reads a stale one."""

    def __init__(self, stale_steps, initial):
        if stale_steps < 0:
            raise ValueError(f'stale_steps must be >= 0, got {stale_steps}')
        self._ring = deque([initial], maxlen=stale_steps + 1)

    def push(self, snapshot):
        self._ring.append(snapshot)

    def stale(self):
        return self._ring[0]


class FinishLine:
    """Counts laps the way f110_env does: the first crossing is the out-lap."""

    def __init__(self, x=FINISH_LINE_X, y_min=FINISH_LINE_Y[0], y_max=FINISH_LINE_Y[1]):
        self.x = float(x)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.crossings = 0
        self.lap_times = []
        self._last_cross_t = None
        # A backwards crossing must be undone by a forward one before it counts.
        self._debt = 0

    @property
    def laps(self):
        return len(self.lap_times)

    def update(self, prev_xy, xy, t):
        """Feed one step of base_link motion. Returns True if a lap was scored."""
        x0, x1 = float(prev_xy[0]), float(xy[0])
        if not self.y_min <= float(xy[1]) <= self.y_max:
            return False
        if x0 >= self.x > x1:
            self._debt += 1
            return False
        if not x0 < self.x <= x1:
            return False
        if self._debt:
            self._debt -= 1
            return False
        self.crossings += 1
        scored = self._last_cross_t is not None
        if scored:
            self.lap_times.append(t - self._last_cross_t)
        self._last_cross_t = t
        return scored


def count_reversals(steers):
    """Sign flips in the command's direction of change; ties are not flips."""
    signs = np.sign(np.diff(np.asarray(steers, dtype=float)))
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


class InstantSteerPlant(plant_sim.GymPlant):
    """Ablation: the judged plant with the steering servo removed.

    Steering snaps to the command every step, so `pid_steer` sees zero error.
    Comparing against the real plant isolates what the servo costs.
    """

    def step(self):
        self.state[plant_sim.DELTA] = np.float32(self.steer_cmd)
        return super().step()


PLANTS = {
    'st': plant_sim.GymPlant,               # the judged plant
    'kinematic': plant_sim.KinematicPlant,  # same servo, no tyre slip
    'instant': InstantSteerPlant,           # same tyres, no servo
}


def make_plant(name, seed=None):
    """Build the plant a `--plant` flag names."""
    if name not in PLANTS:
        raise ValueError(f'unknown plant {name!r}; choose from {sorted(PLANTS)}')
    return PLANTS[name](seed=seed)


def _repo_root():
    node = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        node = os.path.dirname(node)
        if os.path.isfile(os.path.join(node, 'maps', 'icra26.yaml')):
            return node
    raise FileNotFoundError('maps/icra26.yaml not found above ' + __file__)


def _feed_driver(core, scanner, state, t_sim, lr):
    """Hand the driver one snapshot, in the frames the bridge publishes."""
    x, y, yaw = plant_sim.base_link_pose_of(state, lr)
    vx, vy, wz = plant_sim.odom_twist_of(state)
    core.set_odom(x, y, yaw, math.hypot(vx, vy), beta=math.atan2(vy, vx),
                  yaw_rate=wz, t_sim=t_sim)
    ranges = scanner.scan((float(state[plant_sim.X]), float(state[plant_sim.Y]),
                           float(state[plant_sim.PSI])))
    core.set_scan(ranges, scanner.angle_min, scanner.angle_increment, t_sim=t_sim)


class _Log:
    """Per-step and per-command traces, reduced to metrics at the end."""

    def __init__(self):
        self.cmd_steer = []
        self.solve_ms = []
        self.holds = []
        self.saturated = []
        self.delta = []
        self.track_err = []
        self.speed = []
        self.delta_est_err = []
        self.distance = 0.0


def _stat(fn, values):
    """A summary statistic, or 0.0 for a run that logged nothing."""
    return float(fn(values)) if len(values) else 0.0


def _summarise(cfg, log, plant, counter, line, wall_s, contact):
    cmd = np.asarray(log.cmd_steer)
    delta = np.asarray(log.delta)
    err = np.asarray(log.track_err)
    holds = np.asarray(log.holds, dtype=float)
    reversals = count_reversals(cmd)
    return {
        'plant': cfg.plant,
        'seed': cfg.seed,
        'stale_steps': cfg.stale_steps,
        'rtf': cfg.rtf,
        'fixed_solve_ms': cfg.fixed_solve_ms,
        'solve_scale': cfg.solve_scale,
        'sim_time': plant.time,
        'wall_time': wall_s,
        'laps_requested': cfg.laps,
        'laps': line.laps,
        'lap_times': list(line.lap_times),
        'best_lap': min(line.lap_times) if line.lap_times else None,
        'distance_m': log.distance,
        'mean_speed': float(np.mean(log.speed)) if log.speed else 0.0,
        'collisions': counter.events,
        'contact_steps': counter.steps_in_contact,
        'penalty_s': counter.penalty,
        'disqualified': counter.disqualified,
        'contact_model_approximate': contact.approximate,
        'sat_duty': float(np.mean(log.saturated)) if log.saturated else 0.0,
        'cmd_slew_mean': float(np.mean(np.abs(np.diff(cmd)))) if cmd.size > 1 else 0.0,
        'cmd_slew_max': float(np.max(np.abs(np.diff(cmd)))) if cmd.size > 1 else 0.0,
        'actual_slew_mean': (float(np.mean(np.abs(np.diff(delta))))
                             if delta.size > 1 else 0.0),
        'track_err_mean': float(np.mean(err)) if err.size else 0.0,
        'track_err_p95': float(np.percentile(err, 95)) if err.size else 0.0,
        'reversals': reversals,
        'reversals_per_m': reversals / log.distance if log.distance > 0 else 0.0,
        'delta_est_err_mean': _stat(np.mean, log.delta_est_err),
        'delta_est_err_p95': _stat(lambda v: np.percentile(v, 95), log.delta_est_err),
        'delta_est_err_max': _stat(np.max, log.delta_est_err),
        'solve_ms_mean': _stat(np.mean, log.solve_ms),
        'solve_ms_p95': _stat(lambda v: np.percentile(v, 95), log.solve_ms),
        'solve_ms_max': _stat(np.max, log.solve_ms),
        'hold_steps_mean': _stat(np.mean, holds),
        'hold_steps_max': int(holds.max()) if holds.size else 0,
        'dead_time_ms_mean': (cfg.stale_steps + _stat(np.mean, holds)) * PHYSICS_DT * 1e3,
    }


def run(cfg: HarnessConfig):
    """Drive DriverCore against the judged plant until the laps, a DQ, or the limit."""
    track = scan_sim.load_map(os.path.join(_repo_root(), 'maps', 'icra26.yaml'))
    scanner = scan_sim.ScanSimulator(track, cull_reach=SCAN_CULL_REACH)
    contact = scan_sim.ContactResolver(track)
    plant = make_plant(cfg.plant, cfg.seed)
    plant.reset(SPAWN_POSE)
    lr = float(plant.params.lr)
    core = DriverCore(cfg=DriverConfig(),
                      mpc_controller=MPCController(vehicle_model=VehicleModel(), **JUDGED_MPC))

    counter = scan_sim.CollisionCounter()
    line = FinishLine()
    ring = SnapshotRing(cfg.stale_steps, (plant.state.copy(), plant.time))
    log = _Log()
    prev_xy = plant.base_link_pose[:2]
    next_report = 5.0
    wall0 = time.perf_counter()

    while (line.laps < cfg.laps and not counter.disqualified
           and plant.time < cfg.time_limit):
        state, t_snap = ring.stale()
        _feed_driver(core, scanner, state, t_snap, lr)
        # CPU time of this thread, not wall time: other processes on the host
        # must not lengthen the hold. The solver is single-threaded (small
        # SLSQP problem), so thread time is all of its work.
        t0 = time.thread_time()
        cmd = core.step(t_sim=t_snap)
        solve_s = time.thread_time() - t0
        log.solve_ms.append(solve_s * 1e3)
        # The estimate refers to the snapshot's instant, like the pose it rides with.
        log.delta_est_err.append(abs(core.delta_hat - float(state[plant_sim.DELTA])))

        held = hold_for(cfg, solve_s)
        log.holds.append(held)
        log.cmd_steer.append(cmd.steering_angle)

        # The sim keeps running the previous command while this one solves;
        # it reaches the bridge only when the solve ends.
        for _ in range(held):
            plant.step()
            counter.update(plant.time, contact.resolve(plant.state))
            log.saturated.append(plant.is_rate_saturated)
            log.delta.append(float(plant.state[plant_sim.DELTA]))
            log.track_err.append(abs(plant.steer_cmd - log.delta[-1]))
            log.speed.append(float(plant.state[plant_sim.V]))
            xy = plant.base_link_pose[:2]
            log.distance += math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])
            if line.update(prev_xy, xy, plant.time) and not cfg.quiet:
                print(f'  lap {line.laps}: {line.lap_times[-1]:.2f} s')
            prev_xy = xy
            ring.push((plant.state.copy(), plant.time))
            if counter.disqualified or line.laps >= cfg.laps:
                break
        plant.latch(cmd.steering_angle, cmd.speed)
        # Published now, a solve after the scan it was planned from.
        core.note_published(cmd, t_sim=plant.time)

        if not cfg.quiet and plant.time >= next_report:
            next_report += 5.0
            print(f'  t={plant.time:6.1f}s  laps={line.laps}  s={log.distance:6.1f}m  '
                  f'v={log.speed[-1]:.2f}  coll={counter.events}  '
                  f'sat={np.mean(log.saturated):.1%}')

    return _summarise(cfg, log, plant, counter, line, time.perf_counter() - wall0, contact)


def verdict(r):
    """Reasons the controller fails; empty means it passes."""
    reasons = []
    if r['laps'] < r['laps_requested']:
        reasons.append(f"completed {r['laps']}/{r['laps_requested']} laps")
    if r['collisions']:
        reasons.append(f"{r['collisions']} collisions")
    if r['sat_duty'] > MAX_SAT_DUTY:
        reasons.append(f"rate-saturated {r['sat_duty']:.1%} of steps > {MAX_SAT_DUTY:.0%}")
    if r['cmd_slew_mean'] > MAX_MEAN_CMD_SLEW:
        reasons.append(f"cmd slew {r['cmd_slew_mean']:.4f} > {MAX_MEAN_CMD_SLEW}")
    if r.get('delta_est_err_max', 0.0) >= MAX_DELTA_EST_ERR:
        reasons.append(f"wheel estimate off by {r['delta_est_err_max']:.4f} rad "
                       f">= {MAX_DELTA_EST_ERR}")
    return reasons


def collisions_per_10_laps(r):
    return r['collisions'] * 10.0 / max(r['laps'], 1)


def reproduces_judged_failure(r):
    """Whether a run on today's controller fails the way the judged run did."""
    return (collisions_per_10_laps(r) >= GATE_MIN_COLLISIONS_PER_10
            and abs(r['sat_duty'] - LIVE_SAT_DUTY) <= GATE_SAT_TOLERANCE)


def report(r):
    laps = ', '.join(f'{t:.2f}' for t in r['lap_times']) or '-'
    per10 = collisions_per_10_laps(r)
    print(f"\nplant={r['plant']}  stale={r['stale_steps']} steps  rtf={r['rtf']}  "
          f"solve={'measured x' + str(r['solve_scale']) if r['fixed_solve_ms'] is None else str(r['fixed_solve_ms']) + ' ms'}")
    print(f"  sim / wall        {r['sim_time']:.1f} s / {r['wall_time']:.1f} s")
    print(f"  laps              {r['laps']}/{r['laps_requested']}  [{laps}] s")
    print(f"  distance, speed   {r['distance_m']:.1f} m, mean {r['mean_speed']:.2f} m/s")
    print(f"  collisions        {r['collisions']} ({per10:.1f} per 10 laps; "
          f"{r['contact_steps']} contact steps){'  DQ' if r['disqualified'] else ''}")
    print(f"  rate-sat duty     {r['sat_duty']:.1%}  (gate <= {MAX_SAT_DUTY:.0%})")
    print(f"  cmd slew          mean {r['cmd_slew_mean']:.4f}, max {r['cmd_slew_max']:.4f} "
          f"rad/cmd  (old gate < {MAX_MEAN_CMD_SLEW})")
    print(f"  actual slew       mean {r['actual_slew_mean']:.4f} rad/physics step")
    print(f"  |cmd - delta|     mean {r['track_err_mean']:.3f}, p95 {r['track_err_p95']:.3f} rad")
    print(f"  |delta_hat - delta| mean {r['delta_est_err_mean']:.4f}, "
          f"p95 {r['delta_est_err_p95']:.4f}, max {r['delta_est_err_max']:.4f} rad "
          f"(gate < {MAX_DELTA_EST_ERR})")
    print(f"  reversals         {r['reversals']} ({r['reversals_per_m']:.2f} per m)")
    print(f"  solve             mean {r['solve_ms_mean']:.1f}, p95 {r['solve_ms_p95']:.1f}, "
          f"max {r['solve_ms_max']:.1f} ms")
    print(f"  dead time         {r['dead_time_ms_mean']:.0f} ms mean "
          f"(hold {r['hold_steps_mean']:.1f} steps, max {r['hold_steps_max']})")
    print(f"  judged baseline   {JUDGED_BASELINE}")
    fails = verdict(r)
    print(f"  VERDICT           {'FAIL -- ' + '; '.join(fails) if fails else 'PASS'}")
    print(f"  judged failure    {'REPRODUCED' if reproduces_judged_failure(r) else 'not reproduced'}"
          f"  (needs >= {GATE_MIN_COLLISIONS_PER_10} collisions/10 laps and "
          f"sat {LIVE_SAT_DUTY:.1%} +/- {GATE_SAT_TOLERANCE:.0%})")


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--laps', type=int, default=2)
    ap.add_argument('--max-time', type=float, default=None, help='sim seconds')
    ap.add_argument('--stale-steps', type=int, default=DEFAULT_STALE_STEPS)
    ap.add_argument('--rtf', type=float, default=DEFAULT_RTF)
    ap.add_argument('--fixed-solve-ms', type=float, default=None,
                    help='hold every command for this solve time (deterministic)')
    ap.add_argument('--solve-scale', type=float, default=DEFAULT_SOLVE_SCALE,
                    help='multiply measured solve times by this (container vs host)')
    ap.add_argument('--plant', choices=sorted(PLANTS), default='st')
    ap.add_argument('--seed', type=int, default=None,
                    help='plant noise seed; measured-solve runs still vary with host timing')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--json', default=None, help='write the result dict here')
    return ap


def _config_from(args):
    return HarnessConfig(laps=args.laps, max_time=args.max_time,
                        stale_steps=args.stale_steps, rtf=args.rtf,
                        fixed_solve_ms=args.fixed_solve_ms, solve_scale=args.solve_scale,
                        plant=args.plant, seed=args.seed, quiet=args.quiet)


def parse_config(argv=None):
    """Turn command-line flags into a HarnessConfig."""
    return _config_from(_parser().parse_args(argv))


def main():
    args = _parser().parse_args()
    cfg = _config_from(args)
    r = run(cfg)
    report(r)
    if args.json:
        with open(args.json, 'w') as handle:
            json.dump({'config': asdict(cfg), 'result': r}, handle, indent=2)


if __name__ == '__main__':
    main()
