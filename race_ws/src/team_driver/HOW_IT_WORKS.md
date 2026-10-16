# How our driver works, and how we got to 9.85 s

Team CalcuLOL, Track 1 (icra26). This file tells the whole story: where we
started, what went wrong, what fixed it, and what every file in `team_driver/`
is for. It is written so you can follow it without a control-theory
background. Words like "MPC" are explained the first time they appear.

**Where we ended up:** best lap **9.85–9.87 s**, 10 laps in about **98.6–98.8 s**,
**0 collisions** on a quiet machine, in the judge's own `scripts/evaluate.sh`.
That is about 7.5 m/s average round a 74 m lap that has cone slaloms and a
hairpin.

---

## 1. The challenge in one paragraph

The simulator gives the car two things: a LiDAR scan (819 distance readings
over 270°) and its exact position, heading and speed (ground-truth odometry,
which the rules allow). Our program publishes two numbers on `/drive`, 40
times a second: a **steering angle** and a **speed**. The judge counts the
fastest lap and the 10-lap total. Every collision adds 10 s to the lap, and
more than 10 collisions disqualifies the run. So one crash costs far more
than any tuning can win back, and **a clean lap always beats a fast lap that
touches a wall**.

---

## 2. The journey at a glance

Times are judged best laps from our own `evaluate.sh` runs (the files in
`results/`).

| Date | What changed | Best lap | Collisions |
|---|---|---|---|
| 17 Sep | Organisers' template: drives straight into the first wall | — | DQ |
| 26 Sep | Follow-the-gap (reactive, LiDAR only) | ~29.3 s | 0 |
| 29 Sep | First MPC (follows the track centreline) | ~51.7 s | **7** |
| 1 Oct | MPC made stable (Phase 1: faithful simulator, servo model, dead-time fix) | ~50.0 s | 0–1 |
| 2 Oct | Racing line + speed profile + dynamic model + output path unlocked | ~18.0 s | 0 |
| 2 Oct | Pushed the cornering budget with sweeps, slowed only where it crashed | 14.2 s | 0 |
| 2–3 Oct | Racing line reworked for the tightest corner | 12.98 s | 0 |
| 5–6 Oct | MPC allowed to plan acceleration ("reachable pace") + re-tune | 11.08 s | 0 |
| 6 Oct | Offline minimum-time racing line | 10.87 s | 0 |
| 8–9 Oct | Braking limit in corners (g-g), smarter emergency cap, re-refined line | **9.85–9.87 s** | 0 |

The MPC run on 29 Sep was *slower* than the gap-follower. That is normal: the
gap-follower was already tuned, while the MPC was brand new and had problems
nobody could see yet. Fixing them took the whole first week, and every later
gain was built on that.

---

## 3. The story, phase by phase

### Phase 0: a reactive driver (≈29 s)

The first working driver was **follow-the-gap**. It looks at the LiDAR, puts
a "bubble" around the nearest obstacle, finds the widest open gap, and steers
at its middle. It needs no map and no planning, so it was the quickest way to
get round. It completed clean laps at about 29 s.

It can't go much faster, though. It only reacts to what it sees right now, so
it can't brake *before* a corner or pick a good line through it. To go fast
you need to plan ahead using the map, and that is why we moved to MPC.
Follow-the-gap is still in the code today, as a fallback when there is no map
or pose (`DriverCore.plan`).

### Phase 1: making the MPC drive at all (51.7 s with 7 crashes → 50 s clean)

**What MPC is, simply.** Model Predictive Control works like a chess player
who thinks a few moves ahead. Every 25 ms it:

1. takes where the car is now,
2. imagines several different steering and throttle plans for the next half
   second,
3. uses a model of the car to predict where each plan ends up,
4. scores each prediction: distance from the racing line, heading error,
   speed error, and how jerky the steering is,
5. picks the cheapest plan, sends **only its first command**, and throws the
   rest away.

Then it repeats 25 ms later with fresh data. Re-planning all the time is what
makes it robust: a small mistake gets corrected on the next cycle.

**The problem.** Our first MPC finished laps, but the steering chattered
between full left and full right lock (300 direction flips in 700 steps) and
the car hit walls. Tuning the cost weights didn't help. The reason turned out
to be that **our offline test harness was lying to us.** It tested the MPC
against its own perfect model, so it couldn't see what was going wrong in
the real simulator. We found six root causes:

| # | Root cause, simply | What fixed it |
|---|---|---|
| 1 | The MPC's "previous steering" memory was being fed the gap-follower's steering, which jumps around | The MPC keeps its own command memory (Step 4) |
| 2 | The steering wheel isn't instant. The simulator turns it with a motor (a "servo") that needs about 0.26 s for a full left-to-right swing, but the MPC thought it could swing instantly | The servo became part of the MPC's model (Step 6, `ServoModel`) |
| 3 | ~140 ms of **dead time**: by the time a command reached the car, the scan it was based on was old, and the car had moved | Fixed-rate timer, small sensor queues, and later the latch predictor (Step 3, `latch_predictor.py`) |
| 4 | Raw optimiser output went straight to the car: no smoothing, and the wheel snapped to centre on any failure | Output conditioning and smooth coasting (Step 7) |
| 5 | The solver took 30–40 ms on a 25 ms budget, so it often ran out of time | Move blocking + a solve budget (Step 5) |
| 6 | Our tuning file was silently ignored (a ROS naming mistake) | One parameter table for everything (Step 8, `driver_params.py`) |

The most important move was **building a faithful copy of the judge offline**
(`plant_sim.py`, `scan_sim.py`, `lap_sim.py`). It imports the judge's actual
vehicle physics, LiDAR geometry and collision check, instead of re-typing
them. From then on, a result in our harness predicted the judged result to
within about 0.05–0.1 s. Every later improvement was found in that harness
first, then confirmed with `evaluate.sh`.

At the end of Phase 1 the car was stable but slow: 50 s, about 1.5 m/s. Every
limit was still set at a "safe" 0.8 m/s².

### Phase 2: from 50 s to 13 s, by going faster on purpose

Next we worked out how fast the car could physically go, then removed what
was holding it back:

- **A racing line** (`raceline.py`). The centreline is 78.5 m long and has
  sharp kinks. A racing line uses the full track width to make corners
  gentler, and the gentler the corner, the faster you can take it. We first
  built a **minimum-curvature** line from the map.
- **A speed profile** (`speed_profile.py`). It works out the fastest speed at
  every point of the line, including braking early enough for the next corner
  (see section 4.2).
- **A better car model.** At high speed the car slides sideways a little
  (sideslip). The simple "kinematic" model assumes it never slides, which
  becomes wrong above a few m/s. We switched the MPC to the same
  **single-track (ST)** model the judge uses.
- **A longer look-ahead for free.** The MPC still plans 8 steps, but they now
  get longer the further out they are (25, 25, 50, 50, 75, 75, 100, 100 ms).
  That covers 0.5 s, enough to see the next corner, without making the solve
  slower.
- **Unlocked the output path.** The speed we send is turned into acceleration
  by a simple controller inside the simulator. We now invert that controller,
  so "I want 6 m/s² of braking" actually gives 6 m/s². Before this, the car
  got only about 12% of the acceleration the MPC asked for.
- **Sweep, then back off locally.** We raised the cornering budget a lot, ran
  many laps, and recorded *where* each crash happened (its arc position `s`
  along the line). Only those short stretches ("sectors") get a lower budget.
  The rest of the lap stays fast. This took us to 14.2 s.
- **Fixed the tightest corner in the line itself.** One corner (around
  s ≈ 40) was sharper than the steering lock allows, so the car crawled
  through it. The line builder now penalises sharp spots more, which opens
  that corner up. Result: 12.98 s.

### Phase 3: 13 s → 10.87 s

- **The hidden accelerator limit** (`525ce12`). The MPC's "where should I be
  in 0.3 s" target moved at the car's *current* speed. So whenever the car
  tried to speed up, the MPC saw it as running ahead of the target and pushed
  back. Now the target moves at the speed the car *can reach* (accelerating
  as hard as the plant allows, braking at our limit). That one fix closed most
  of the gap between the planned and the driven lap time. With a re-tune:
  11.08 s.
- **A minimum-time line** (`c15552a`). Minimum curvature is only a stand-in
  for "fastest". The line is now refined offline to minimise lap time
  directly, inside the same wall clearances. It ships as
  `config/icra26_raceline.csv`, and the driver checks it against the map
  before using it. Result: 10.87 s.

### Phase 4: under 10 s (9.85–9.87 s)

Above a cornering budget of about 18 m/s² the car started spinning in a few
places. The investigation found **why**:

- **Braking into a corner makes the car unstable.** When you brake, weight
  moves to the front wheels and the rear wheels lose grip. In the judge's
  physics, the car's rotation becomes unstable under hard braking above about
  6 m/s. The old profile braked at a full 6 m/s² deep into corners, so the
  rear stepped out and the car spun into the wall.
- **Fix: the g-g limit** (`47eefc7`). Braking and cornering now share one
  budget, shaped like an ellipse. The harder a corner already turns, the less
  braking is allowed in it. A real driver does the same: brake in a straight
  line, then turn. This removed the spins.
- **The emergency cap was false-alarming** (`42cb78f`). The safety check
  looked straight ahead past the end of the 0.5 s plan, saw the outside wall
  of the next corner, and slowed the car for no reason. It now follows the
  racing line round the corner, the same way the car will.
- **Re-refined the line at the new profile and pinned the settings**
  (`186026c`): cornering budget 21 m/s², two short sectors slowed slightly
  (s 3–7 at 95%, s 43–47 at 85%), g-g braking on, and the improved emergency
  cap on.

We picked **21** rather than the faster 22 (9.71 s in the harness) on
purpose. 22 crashed in 2 of 18 runs when the computer was busy. 21 crashed in
0 of 24. One crash costs +10 s, so 0.15 s isn't worth that risk.

---

## 4. The core algorithm

### 4.1 The big picture

```
 OFFLINE (once, before the race)          ONLINE (every 25 ms, 40 Hz)
 ───────────────────────────────          ─────────────────────────────────────────
 map ─► raceline.py ─► icra26_raceline.csv ─► check against map ─► reference line
                                                                       │
                                     speed_profile.py ◄────────────────┘
                                     (fastest safe speed at every point)
                                                                       │
  /ego_racecar/odom ─► latch_predictor ── "where will the car be when ──┤
  (pose, speed,        (replays the last    this command takes effect"  │
   sideslip, yaw rate)  commands through                                ▼
                        the judge's physics)                  MPC  (mpc_controller.py)
                                                              8 steps, 0.5 s ahead,
                                                              ST car model + servo model,
                                                              SLSQP optimiser, 15 ms budget
                                                                       │ steer, accel
  /scan ─► preprocess ─► emergency speed cap  ─────────────────────────┤
                         (can we still stop                            ▼
                          before the wall?)                    output conditioning
                                                               (slew limit, profile cap,
                                                                emergency cap, speed-loop
                                                                inversion)
                                                                       │
                                                                       ▼
                                                              /drive (steering, speed)
```

### 4.2 Step by step

**1. Reference line (offline + startup check).** `raceline.py` reads the map
image, seals the cone rows so no line can squeeze between cones (that's
against the rules), and measures how far the car can move sideways at each
point before getting too close to a wall (half the car width plus a 0.25 m
margin). It then finds the smoothest line inside those limits (minimum
curvature, a least-squares problem), and refines it offline for lap time. At
startup `driver_core` loads the CSV and **checks every point against the map**.
If the check fails, it builds a minimum-curvature line from the map instead,
or falls back to the centreline.

**2. Speed profile.** For every point on the line:

- *Cornering limit*: `v = sqrt(a_lat_max / curvature)`. A tighter corner
  gives a lower speed. `a_lat_max` (21 m/s²) is how much sideways
  acceleration we allow.
- *Steering lock limit*: in hairpins, the speed at which the wheels would
  have to turn past their maximum angle.
- *Braking pass* (run backwards round the lap): make sure the car can slow
  down in time for every corner, using the g-g ellipse from Phase 4.
- *Acceleration pass* (run forwards): make sure the car can actually speed up
  that fast.

The lap is closed, so both passes wrap round: the corner right after the
finish line is braked for too.

A detail about the judge's physics: its tyres are "linear", meaning they never
lose grip suddenly the way real tyres do. So `a_lat_max` isn't really a grip
limit. It's a **slip budget**: how far we let the car slide before it gets
hard to control.

**3. Latch prediction (the dead-time fix).** When the driver reads a pose, it
is already about 100 ms old, and our command reaches the car one solve later.
At 7 m/s that's more than half a metre the MPC would never know about. But we
know every command we sent and when each one took effect. So
`latch_predictor.py` **replays those commands through the judge's own physics
equations**, starting from the measured state, and works out where the car
will be when the new command takes effect. The MPC plans from *that* point.

**4. The MPC itself** (`mpc_controller.py`):

- *State* (what the model tracks): position x and y, heading, speed,
  sideslip, wheel angle, and yaw rate.
- *Model*: the judge's single-track physics (`vehicle_model.py`), with the
  steering servo built in. So the MPC knows the wheel can only move about
  0.08 rad per 25 ms.
- *Horizon*: 8 steps covering 0.5 s, with step lengths growing from 25 ms to
  100 ms.
- *Decisions*: only 3 distinct moves of (steer, accel), held over blocks of
  steps (`block_map [0,1,1,2,2,2,2,2]`). So the optimiser has only 6 numbers
  to choose. This **move blocking** is what makes it fast enough.
- *Cost*: position error × 2, heading error × 2, speed error × 0.5, steering
  effort × 0.08, steering change × 3.0 (this weight stops the chatter), plus
  small acceleration terms.
- *Target*: points along the racing line, spaced at the speed the car can
  actually reach (the Phase 3 fix).
- *Solver*: SciPy's SLSQP. It starts from the last plan shifted one step
  (a "warm start"), is capped at 25 iterations, and has a hard 15 ms
  wall-clock budget. If it runs out of time, the car keeps following the
  previous plan instead of doing something random.

**5. Safety layer and output conditioning** (`driver_core._condition`):

- *Steering slew limit*: never ask the wheel to move faster than the servo
  can. This removes only motion the wheel couldn't have followed anyway, so
  it adds no lag.
- *Acceleration clamp*: the MPC's request is limited to braking at
  6 m/s² and accelerating at 9.51 m/s².
- *Profile cap*: the speed at the end of this 25 ms step may not exceed the
  speed profile.
- *Emergency cap*: from the LiDAR, measure the free distance along the path
  the car is about to drive. Limit the speed to one the car could still stop
  from (braking at 7 m/s²). On a clean lap it should essentially never
  trigger. It's the net under the trapeze.
- *Speed-loop inversion*: `/drive` takes a speed, not an acceleration. We
  calculate exactly which speed makes the simulator's internal controller
  produce the acceleration we want.
- *Coasting*: if anything is unusable (no scan yet, a NaN, a solver failure),
  the wheel unwinds smoothly and the car slows down. It never snaps.

### 4.3 Why MPC, and not something else?

| Approach | Why we didn't stop there |
|---|---|
| **Follow-the-gap** (what we started with) | Purely reactive: it can't brake before a corner or choose a line. It topped out around 29 s. |
| **Pure pursuit / Stanley** (steer at a point ahead on the line) | Simple and fast to run, but it doesn't know about the slow steering servo, sideslip or the dead time. At 7–10 m/s through slaloms, those are exactly what decides whether you hit a cone. Every corner would need hand-tuned look-ahead distances. |
| **Reinforcement learning / imitation** | Needs a lot of training time and a reward that doesn't learn to cheat. It's hard to explain why it does what it does, and it's hard to guarantee zero crashes. We'd have to defend it at the interview without being able to point at a reason. |
| **MPC** (chosen) | It uses everything the rules give us: the map, the exact pose and the known physics. It respects the car's real limits (steering lock, servo speed, braking stability) *while* planning, instead of finding out afterwards. It plans 0.5 s ahead, so it brakes before corners. Every decision can be explained by its cost. And because we can import the judge's physics, its model is unusually accurate. |

The main cost of MPC is computing time: it has to solve an optimisation
problem 40 times a second in Python. Move blocking, the stretched time steps
and the 15 ms budget are how we fit it in.

---

## 5. Every file in `team_driver/`, and why it is there

The organisers' template gave us `driver.py` (which only drove straight),
`setup.py`, `setup.cfg`, `package.xml`, `launch/driver.launch.py`, an empty
`config/driver_params.yaml`, `requirements.txt` and `apt-packages.txt`.
Everything else is ours.

### Files that run during the race

| File | What it does | Why it's needed |
|---|---|---|
| `team_driver/driver.py` | The ROS 2 node. It declares parameters, subscribes to `/scan` and `/ego_racecar/odom`, runs the control step on a fixed 25 ms timer, and publishes `/drive`. It contains **no algorithm**. | The judge runs `ros2 run team_driver driver`. Keeping it to wiring means the offline harness runs the exact same algorithm. |
| `team_driver/driver_core.py` | The algorithm, with no ROS code in it: loads and checks the reference line, runs latch prediction, calls the MPC, applies the safety caps and output conditioning, and coasts when needed. Also contains `DriverConfig`, the steering estimator (`SteeringEstimator`) and the latch clock. | Early on, the node and the harness each had their own copy of the control path. The copies drifted apart, so our tests weren't testing what the judge ran. Now there is one path. |
| `team_driver/driver_params.py` | One table of every parameter: its ROS name and default value. It declares, reads, validates and echoes them. | The old YAML block was silently ignored by ROS. With one table, a missing params file is provably harmless, and the startup log shows where the values came from (`params_source`). |
| `team_driver/mpc_controller.py` | The MPC: target window along the line, prediction, cost, SLSQP solve with warm start and time budget, fallback to the previous plan. | The brain that chooses steering and acceleration. |
| `team_driver/vehicle_model.py` | Car models for the MPC: the judge's single-track (ST) physics with exact integration of the sideways motion, the simple kinematic model, the steering-servo model and the simulator's speed loop. | The MPC is only as good as its prediction. These models match the judge's physics (`model_error.py` measures how closely). |
| `team_driver/speed_profile.py` | The fastest safe speed at every point: corner limit, steering-lock limit, braking pass (with the g-g ellipse), acceleration pass, slow-down sectors. | Tells the MPC how fast to go, and caps the speed so the car never enters a corner too fast. |
| `team_driver/latch_predictor.py` | Replays recently sent commands through the judge's servo, speed loop and physics to predict the car's state when the next command takes effect. | Removes the ~100–150 ms dead time that once made the car oscillate. |
| `team_driver/longitudinal.py` | Geometry for the speed caps: positions along the closed racing line by distance, and the free distance in a car-wide corridor along the planned path, from the LiDAR. | Lets the emergency cap look *along the turn* rather than straight at the outside wall. |
| `team_driver/raceline.py` | Builds the racing line from the map: seals the cones, measures wall clearance, solves for minimum curvature, and refines for minimum lap time offline (`--refine --write`). | The racing line is the biggest single source of lap time. It's built from the map by our own code, as rule 32 requires. |
| `config/icra26_raceline.csv` | The precomputed minimum-time racing line (743 points, 74.3 m, 0.25 m wall margin). | Refining takes about 1–3 minutes, too slow for startup. The driver checks it against the map before trusting it. |
| `config/driver_params.yaml` | Every tunable value, kept equal to the code defaults (a test enforces this), with comments saying why each value was chosen. | Lets us retune without editing code, and records the reasoning. |
| `requirements.txt` | Pins `numpy==2.5.3` and `scipy==1.18.1`. | SciPy provides the SLSQP solver and the least-squares solver for the racing line. Pinned versions mean the judge builds what we tested. |

### Offline tools (never run during the race)

| File | What it does | Why it's needed |
|---|---|---|
| `team_driver/plant_sim.py` | The judge's car physics, run offline. It *imports* the simulator's own code from `external/f1tenth_gym` (with small stubs for packages that aren't installed), rather than copying it. | The old harness used our own model and missed the steering servo completely, so every tune against it was blind. Importing the judge's code means it can't drift from the judge. |
| `team_driver/scan_sim.py` | The judge's LiDAR and wall-collision check, offline: the judge's own wall extraction, a line-by-line port of its ray-casting, its car-body rectangle, and the referee's collision counting rule (at most one per second). | Lets the harness see the walls, test the LiDAR-based safety layer, and count collisions the way the referee does. |
| `team_driver/lap_sim.py` | The closed-loop test harness: runs `DriverCore` against `plant_sim` and `scan_sim` with the judge's timing (stale sensor data, solve time delaying the command, the 40 Hz timer). Supports overrides (`--driver KEY=VALUE`, `--mpc KEY=VALUE`), fixed or measured solve times, and JSON output. | This is where every improvement was found and checked across a grid of timing conditions before a judged run. |
| `team_driver/lap_floor.py` | Prints the ideal lap time for a line at different cornering budgets, assuming a perfect driver. | Shows how much time is physically available, and how far the real car is from it (tracking loss). |
| `team_driver/model_error.py` | Measures how far each MPC model's prediction drifts from the judge's physics, at different speeds. | Evidence for choosing the ST model over the kinematic one. |
| `team_driver/mpc_controller_simple.py` | An early grid-search prototype of the MPC. **Nothing imports it.** | Kept from the first day for reference only. It can be deleted. |
| `.coverage` | A test-coverage data file. | A build artefact that got committed by accident. It is safe to delete. |

### Tests (`test/`, 27 files, ~660 tests, ~95% coverage)

Grouped by what they protect:

- **Judge fidelity**: `test_plant_sim`, `test_scan_sim`, `test_st_model`,
  `test_steering_actuator`, `test_model_error`. The offline physics, LiDAR,
  servo and ST model match the judge's code.
- **Harness behaviour**: `test_lap_sim`, `test_harness_cadence`,
  `test_harness_overrides`, `test_dead_time`, `test_dt_schedule`. Timing,
  staleness, overrides, and runs that repeat exactly.
- **MPC behaviour**: `test_command_memory`, `test_move_blocking`,
  `test_fused_objective`, `test_reference_wrap`, `test_reference_pace`,
  `test_reference_speed_profile`, `test_latch_predictor`, `test_speed_loop`.
- **Line and profile**: `test_raceline`, `test_speed_profile`,
  `test_speed_profile_physics` (the g-g braking).
- **Safety and wiring**: `test_output_conditioning`, `test_step4_safety`,
  `test_cap_reference`, `test_step3_wiring`, `test_driver_params` (the YAML
  matches the code, and bad values are refused).

Run them with `python3 -m pytest test/` from `race_ws/src/team_driver`.
**Never run them during `evaluate.sh`**: the extra CPU load slows the solver
and causes crashes (see section 6).

---

## 6. Things we learned the hard way

- **A test harness that can't see the problem is worse than no harness.**
  Days of weight tuning did nothing until the harness used the judge's own
  physics and timing.
- **Robustness beats the last tenth of a second.** Every setting was checked
  over a grid: stale data at 5, 9 and 14 physics steps, fixed solve times of
  5, 7 and 12 ms, measured solve times, and 10-lap runs. A setting had to be
  clean across all of them before it was pinned.
- **Fixed solve times hide crashes.** Some crashes only appear when the solve
  time varies the way it does on a real CPU. Collisions were screened with
  measured timing, and lap time was ranked with fixed timing.
- **A busy computer causes crashes.** Three judged runs were disqualified
  because tests were running on the same machine at the same time. When the
  CPU is overloaded, solves overrun their 15 ms budget and the car falls back
  to old plans. The first corner is the most sensitive spot. Check `uptime`
  before a judged run.
- **Things that didn't work**, so don't retry them: braking harder than
  6 m/s² (the car spins); a cornering budget of 23 (crashes at s ≈ 54);
  slow-down sectors that are too gentle (0.8 at s 62–68 and 0.9 at s 36–43
  each collided under measured timing); raising the MPC horizon to look
  further ahead (solve time grows much faster than the horizon). Also, a wall
  margin of 0.20 m instead of 0.25 m brought a collision back.
- **Headless runs are about 0.01–0.02 s per lap slower than runs with the
  GUI.** The judges run headless, so plan on ~9.87 s. The cause is still
  unknown. A harness test ruled out the simulator's speed difference
  (real-time factor) as the reason.

---

## 7. Reproducing the result

```sh
# inside the container
cd /hackathon/race_ws && colcon build --symlink-install && source install/local_setup.bash

# judged-style run (quiet machine!)
cd /hackathon && ./scripts/evaluate.sh --team CalcuLOL --headless

# offline harness: 10 laps under judged timing
cd /hackathon/race_ws/src/team_driver/team_driver
python3 lap_sim.py --laps 10 --json /tmp/r.json

# regenerate the racing line (single-threaded, or the line shifts by ~9 cm)
OMP_NUM_THREADS=1 python3 raceline.py --margin 0.25 --refine --write ../config/icra26_raceline.csv
```

Fallbacks if a judged run ever collides: cornering budget 22 with s 3–7 at
0.85 (9.71 s in the harness, but less margin), or the previous line and
profile at commit `42cb78f` (10.81 s).
