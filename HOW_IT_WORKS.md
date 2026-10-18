# How our driver works

Team CalcuLOL, Track 1 (`icra26`). This document explains the whole driver in
`race_ws/src/team_driver/`: what every part does, why it is built that way,
what we tried that failed, where it can still break, and how we know. It is
written for someone with no robotics or control background. Every technical
word is explained the first time it appears, and there is a glossary at the
end (section 15).

**Where we ended up (final car, commit `ac836d8`):**

| | |
|---|---|
| Best lap | **8.038–8.047 s** |
| 10-lap total | **80.49–80.55 s** |
| Collisions | **0**, including the out lap and the warm-up lap |
| Runs | 10 of 10 `COMPLETE` |
| Where measured | The judge's own `scripts/evaluate.sh --headless`, batch `CalcuLOL__20261018T004950Z_r1..r10`, our laptop, on mains power |
| Real-time factor | 2.28–2.35 (the simulator ran about 2.3× faster than real time) |

That is about 9.25 m/s on average round a 74.4 m racing line, with a top
speed of 12 m/s.

**The whole record of the final car code** (`ac836d8`; later commits are
comments and documentation only):

| | |
|---|---|
| Runs on our machine | 30 |
| `COMPLETE` | 30 of 30 |
| Collisions in timed laps | 0 in 300 |
| Runs with any collision | 1 of 30: `unnamed_team__20261018T082903Z_r6` had 4 collisions **before** the timed laps (no time penalty, but they count towards the 10-collision limit) |
| Submitted runs | `results/submitted/CalcuLOL__20261018T102653Z_r1..r3`: 3/3 clean, 8.036–8.045 s, 80.50–80.51 s |

**How to read this.** Sections 1–3 give the background: the challenge, the
physics of the simulated car, and the big picture. Sections 4–6 go through the
driver part by part, then trace one control cycle from start to finish.
Sections 7–11 cover the history, testing, failures, risks and rules. Sections
12–15 list the files, how to reproduce everything, notes for readers of the
code, and a glossary.

---

## Contents

1. [The challenge, and the rules that shaped our design](#1-the-challenge-and-the-rules-that-shaped-our-design)
2. [The simulated car: the physics you need](#2-the-simulated-car-the-physics-you-need)
3. [The big picture](#3-the-big-picture)
4. [Every part of the driver, in detail](#4-every-part-of-the-driver-in-detail)
5. [One control cycle, step by step](#5-one-control-cycle-step-by-step)
6. [Every tuned number, and why it has that value](#6-every-tuned-number-and-why-it-has-that-value)
7. [The journey: from a car that crashed to 8.04 s](#7-the-journey-from-a-car-that-crashed-to-804-s)
8. [How we test and measure](#8-how-we-test-and-measure)
9. [Things we tried that did not work](#9-things-we-tried-that-did-not-work)
10. [Known limitations, risks and failure modes](#10-known-limitations-risks-and-failure-modes)
11. [The rules: how we comply](#11-the-rules-how-we-comply)
12. [Every file, and why it exists](#12-every-file-and-why-it-exists)
13. [Reproducing everything](#13-reproducing-everything)
14. [Things in the code that may look wrong](#14-things-in-the-code-that-may-look-wrong)
15. [Glossary](#15-glossary)

---

## 1. The challenge, and the rules that shaped our design

### 1.1 What the car gets and what it sends

The simulator (`f1tenth_gym`, behind a ROS 2 bridge) gives our program:

| Input | What it is | How we use it |
|---|---|---|
| `/ego_racecar/odom` | The car's exact position, heading, speed, sideways slip and turning rate. It is the simulator's own truth, with no noise. | Everything: it is where the planner starts from. |
| `/scan` | A LiDAR (laser range finder): 819 distance readings spread over 270°, up to 25 m. | A safety net only: an emergency speed limit, plus a fallback driver if the planner fails. |
| `/map` / map files | A picture of the track (17 × 18.2 m, 5 cm per pixel) and a centreline CSV. | Computing the racing line. |

It sends back one message on `/drive`, 40 times per second: a **steering
angle** (radians) and a **speed** (m/s). We do not send an acceleration or a
throttle. The simulator turns our speed into acceleration with its own
controller (section 2.3), and that matters a lot.

**ROS 2** is the messaging framework all of this runs on. Programs ("nodes")
publish and subscribe to named "topics". Our node is `team_driver/driver`.

### 1.2 How the race is run and scored

From `docs/05-evaluation.md` and `docs/06-rules.md`:

- A run is an **out lap** (grid to start line, not timed), **one warm-up lap**
  (not scored), then **10 timed laps** back to back.
- Score = 50 × (best lap of anyone ÷ our best lap) + 50 × (best 10-lap total of
  anyone ÷ our 10-lap total). Both halves count equally, so consistency matters
  as much as one fast lap.
- **Every collision adds 10 s** to the lap it happens on. **More than 10
  collisions disqualifies the run** (scores zero).
- Collisions are counted at most once per second, but **staying against a wall
  keeps counting**: a car stuck on a wall is disqualified after about 11 s.
- **Collisions on the out lap and warm-up lap carry no time penalty but do
  count towards the 10-collision disqualification** (rule 18). This is why we
  built a careful warm-up mode (section 4.10).
- The judges run each entry **three times and keep the best**.
- All timing is in **simulated seconds**, not wall-clock seconds. A slow
  computer does not make the lap slower on paper, but it can still make a slow
  program react late.

The consequence that shaped everything: **one collision costs 10 s, and our
whole lap is 8 s.** A clean lap at 8.1 s beats an 8.0 s lap with a touch. So
every speed gain had to be proven safe across many conditions before we kept
it.

### 1.3 What we are allowed to use

- Ground-truth odometry is **explicitly allowed** (rule 31). We use it, and we
  did not build our own localisation.
- Offline precomputation is allowed (rule 32), **if it is produced by code we
  submit and is reproducible from the map alone.** Hand-typed waypoints are
  not allowed. Our racing line is computed by `raceline.py` from the map
  (section 4.3).
- **Cone rows are walls** (rule 11). Driving through the gaps between cones is
  not allowed, even though the gaps are wide enough. Our line builder seals
  those gaps before it plans anything.
- We must not touch the judging environment (rule 25): the referee, `docker/`,
  `maps/`, `scripts/`, `install/`, `.github/workflows/` and `external/`. All
  our work is inside `race_ws/src/team_driver/`.
- AI assistants are allowed (rule 36), but every member must be able to
  explain any part of the code (rules 37–38).

---

## 2. The simulated car: the physics you need

You cannot understand our design choices without these six facts about the
simulated car. We found most of them by reading the simulator's source code in
`external/f1tenth_gym` and measuring against it. Every number here comes from
that code or from `VehicleParameters` in `vehicle_model.py`.

### 2.1 The car

| Quantity | Value |
|---|---|
| Wheelbase (front axle to rear axle) | 0.33 m |
| Width × length | 0.31 m × 0.58 m |
| Mass | 3.74 kg |
| Steering lock (maximum wheel angle) | ±0.4189 rad (±24°) |
| Fastest the wheel can turn | 3.2 rad/s |
| Maximum acceleration or braking | 9.51 m/s² |
| Physics step | 10 ms (100 times per second) |

The position the simulator reports (`base_link`) is the **rear axle**, not the
centre of the car. The LiDAR sits 0.275 m in front of it. The front bumper is
0.455 m in front of it.

### 2.2 Fact 1: the steering wheel is slow (the "servo")

When we ask for a steering angle, the wheel does not jump there. A small motor
(a **servo**) turns it, using this rule every 10 ms physics step:

```
turning speed = clip( 38.2 × (asked angle − current angle),  −3.2, +3.2 ) rad/s
```

- For small changes the wheel closes the gap quickly, with a time constant of
  about 26 ms.
- For big changes it hits the **rate limit** of 3.2 rad/s, so it can move only
  **0.08 rad per 25 ms** control step.
- Going from full left to full right lock (0.84 rad) takes **262 ms**. Our
  planner looks only 500 ms ahead, so a full reversal uses half of it.

**Why it matters:** a planner that thinks the wheel moves instantly will ask for
moves the wheel cannot make, then over-correct, and the steering starts to
oscillate. That was our first big problem (section 7, Phase 1). Our fix was to
put the servo inside the planner's model (section 4.6).

### 2.3 Fact 2: we send a speed, but the car responds to an acceleration

The simulator turns our requested speed into an acceleration with a simple
**proportional controller** (it pushes harder the further the car is from the
asked speed):

```
acceleration = gain × (asked speed − current speed)
gain = 4.755 per second when speeding up, 19.02 per second when slowing down
```

That is then clipped to ±9.51 m/s². Above 7.319 m/s there is also a **power
limit**: the most acceleration available is 9.51 × 7.319 / speed. At 12 m/s
that is only 5.8 m/s².

**Why it matters:** if you want 9 m/s² of braking at 10 m/s, sending "10 − a
bit" does almost nothing. You have to send the exact speed that makes this
controller produce 9 m/s² over the next 25 ms. Our code inverts the formula: it
sends **9.45 m/s** for that case (`SpeedLoop.command` in `vehicle_model.py`).
Before we did this, the code sent `current speed + acceleration × 0.025 s`.
With the simulator's gain of 4.755 /s, that delivered only 4.755 × 0.025 ≈ 12%
of the acceleration the planner asked for when speeding up.

### 2.4 Fact 3: the tyres never "lose grip"

The simulator uses a **single-track model** (a "bicycle model": the two front
wheels are treated as one, and so are the two rear wheels) with **linear
tyres**. A linear tyre's sideways force keeps growing with its slip angle
forever. Real tyres reach a peak and then slide.

So in this simulator there is no grip limit. You could in principle corner at
any speed, but the car slides more and more sideways (its **slip angle**
grows), and at some point it can no longer be controlled. The rear slip angle
works out to:

```
rear slip (radians) = sideways acceleration / 56.1
```

This is why our cornering setting `a_lat_max` = 27 m/s² (about 2.75 g, which no
real small car could do) is called a **slip budget**, not a grip limit. At
27 m/s² the rear tyres slip about 27.6°. The car is effectively drifting
through corners.

### 2.5 Fact 4: braking in a corner makes the car spin

When the car brakes, weight shifts forward. In the simulator's equations this
**unloads the rear tyres** (the `h` terms in `_SingleTrackConstants.lateral`).
Under hard braking above a certain speed, the car's rotation becomes
**unstable**: a small yaw (rotation) grows by itself and the rear steps out.

The critical speeds, from the eigenvalues of the simulator's slip and yaw
equations, are about **6.2 m/s while braking at 6 m/s²**, **4.95 m/s at
8 m/s²** and **4.5 m/s at 9 m/s²** (our maximum braking). The car spends most
of the lap above these speeds.
So braking hard *while turning* spins the car, the way a real car spins if you
slam the brakes mid-corner. This one fact caused most of our crashes on the way
from 11 s to 8 s. Our fixes are the g-g braking limit (section 4.4) and letting
the planner own the braking (section 4.6).

### 2.6 Fact 5: timing and dead time

- Physics runs in 10 ms steps. Our driver plans every **25 ms of simulated
  time** (40 Hz), so each command is held for 2 or 3 physics steps.
- In judged runs the simulator runs **as fast as the computer allows**. On our
  laptop that is 2.3× real time. It does not wait for us.
- The pose and scan we receive are **already old**. We measured about 9
  physics steps (90 ms) on 30 Sep, when the node still used depth-10 message
  queues. The node now uses depth 1, and the harness tests 5, 9 and 14 steps.
- Our command also takes effect later than the moment we read the clock. While
  we compute it (a few ms of wall-clock time), the simulator keeps running, so
  the command lands about `solve time × real-time factor` later in simulated
  time.

Together this is called **dead time**: the gap between "the moment the data
describes" and "the moment our command acts". At 9 m/s, 100 ms of dead time is
almost a metre of travel the planner would not know about. Our fix is the
**latch predictor** (section 4.5).

### 2.7 Fact 6: the track

- The racing line is **74.4 m** long. The organisers' centreline is 78.5 m.
- The car goes **counter-clockwise**. The start line is at x = 1.32 m and the
  grid is 3 m behind it.
- Positions along the lap are measured as **arc length `s`** in metres from
  the start of our racing line (about the start line). We name corners by `s`.

The main features on our line (computed from the shipped line, curvature
κ = 1 / radius):

| Where (s, m) | What it is | Tightest radius | Profile speed there |
|---|---|---|---|
| 3–7 | First corner after the start straight (left) | 1.6 m | 6.4 m/s |
| 8–21 | Right-hand corner complex | 1.6 m | 7.1 m/s |
| 28–43 | Hairpin complex: left at s ≈ 34, then the **tightest corner of the lap**, right at s ≈ 42 | 1.0 m | 5.5 m/s |
| 43–47 | The exit of that complex (we call it the "chicane exit") | 2.4 m | 7.0 m/s |
| 49–57 | Fast left kink (the s ≈ 54 corner) | 3.1 m | 9.6 m/s |
| 57–74 | The fastest section, back to the line. Top speed 12 m/s at s ≈ 59 | 4.3 m | 11–12 m/s |

---

## 3. The big picture

### 3.1 The pipeline

```
 OFFLINE (once, ~4 minutes, by us)        AT STARTUP (once, a few seconds)
 ─────────────────────────────────        ─────────────────────────────────
 map image + centreline                    load config/icra26_raceline.csv
   │                                       check every point is ≥ 0.405 m
   ▼                                       from the (cone-sealed) walls
 raceline.py --refine --write              │   fails? → build min-curvature line
   seal cone rows                          │            from the map instead
   min-curvature line (a QP)               ▼
   refine for minimum lap time       speed profile: fastest safe speed at
   │                                 every point (speed_profile.py)
   ▼                                       │
 config/icra26_raceline.csv ───────────────┘

 EVERY 25 ms OF SIMULATED TIME (driver_core.DriverCore.step)
 ────────────────────────────────────────────────────────────────────────────
  odom (pose, speed, slip, yaw rate; ~0.1 s old)
     │
     ▼
  1. estimate where the steering wheel is (SteeringEstimator)
  2. predict where the car WILL BE when this command takes effect,
     by replaying our own recent commands through the simulator's
     equations (LatchPredictor)
     │
  3. pinned against a wall? → reverse off it, skip the rest (unstick)
     │
     ▼
  4. MPC (mpc_controller.py): try steering + acceleration plans over
     the next 0.5 s with a model of the car, pick the cheapest
     │  steer, accel, and its predicted path
     ▼
  5. safety caps: speed profile cap, LiDAR emergency cap along the
     predicted path (longitudinal.py)
     │
     ▼
  6. output conditioning: limit steering change to what the servo can
     follow, turn the wanted acceleration into the exact /drive speed
     │
     ▼
  /drive (steering angle, speed)
```

### 3.2 The one-sentence version

We compute a fast racing line offline from the map. At startup we work out the
fastest safe speed at every point of it. Then, 40 times a second, a **Model
Predictive Controller** plans the next half-second of steering and
acceleration, starting from where the car will actually be when the command
lands, using the simulator's own physics, with a LiDAR safety net underneath.

### 3.3 What MPC is, in plain words

**Model Predictive Control (MPC)** works like a chess player who thinks a few
moves ahead, then re-thinks after every move. Every 25 ms it:

1. takes the car's (predicted) current state;
2. considers a short plan of steering and throttle for the next 0.5 s;
3. uses a model of the car to predict where that plan would take it;
4. scores the prediction: distance from the racing line, heading error, speed
   error, and how much and how suddenly it steers;
5. adjusts the plan to lower that score (this is the "optimisation");
6. sends **only the first command** of the best plan and throws the rest away.

It repeats 25 ms later with fresh data. Re-planning all the time is what makes
it robust: a small error is corrected on the next cycle.

### 3.4 Why MPC, and not something simpler?

| Approach | Why it was not enough |
|---|---|
| **Follow-the-gap** (steer at the widest open gap in the LiDAR; we started here) | Purely reactive: it cannot brake *before* a corner it cannot see yet, or pick a line. It topped out around 28–30 s. It is still in the code as a fallback. |
| **Pure pursuit / Stanley** (steer towards a point some distance ahead on the line) | We built Pure Pursuit: 17.83 s, also tried on a precomputed racing line. It is simple and fast, but it knows nothing about the slow servo, the sideways slip, the dead time or braking instability. At 9–12 m/s those decide whether you hit a wall. Every corner would need hand-tuned look-ahead distances, and its gains were shrinking. |
| **Reinforcement learning / imitation learning** | Needs lots of training and a reward that does not learn to cheat. It is hard to guarantee zero collisions or explain a specific decision. |
| **MPC (chosen)** | It uses everything the rules give us: the map, the exact pose and the known physics. It respects the car's real limits (steering lock, servo speed, braking) *while* planning. It looks 0.5 s ahead, so it brakes before corners. Every decision can be explained by its cost function. And because we can import the judge's physics, its model is unusually accurate. |

The main cost of MPC is computing time: it solves an optimisation problem 40
times a second, in Python. Move blocking, the stretched time steps and the
15 ms budget (section 4.6) are how it fits.

### 3.5 Design principles we stuck to

1. **One control path.** The algorithm (`driver_core.py`) has no ROS code in
   it. The ROS node (`driver.py`) only wires messages in and out, and the
   offline test harness (`lap_sim.py`) drives the *same* `DriverCore` object.
   We learnt this the hard way: an earlier harness had its own copy of the
   control code, which drifted. It fed the planner a different steering value
   than the real node did, so our tests were testing a different car.
2. **Copy nothing from the judge; import it.** The offline simulator imports
   the judge's own vehicle code, wall extraction and collision geometry instead
   of re-typing them, so it cannot silently drift.
3. **Find the root cause before tuning.** When the car crashed, we instrumented
   the crashes until we knew *why*, then fixed the cause (for example: "every
   crash above 23 m/s² is braking yaw", section 7 Phase 6).
4. **A grid of conditions, not one run.** A change was kept only if it was
   clean across many timings, delays and loads (section 8).
5. **Every behaviour change has a switch whose "off" setting is the old car,
   exactly.** For example `gg_brake`, `mpc_full_brake`, `cap_follows_reference`,
   `warmup_crossings = 0` and `unstick_after = 0`. Tests check that "off"
   reproduces the old behaviour bit for bit. This makes A/B comparison and
   rollback instant.
6. **Validate inputs at the boundary, fail safe.** Every config value is
   checked at startup. The shipped racing line is used only if it passes a
   clearance check against the map. Any non-finite (NaN) number makes the car
   coast smoothly rather than do something random.
7. **A fallback at every layer** (section 10.3).

---

## 4. Every part of the driver, in detail

### 4.1 The ROS node (`driver.py`)

This is the entry point the judges run (`ros2 run team_driver driver`). It does
four things and nothing else:

1. **Declares and validates every parameter**, from one table
   (`driver_params.py`, section 4.2). It forces `use_sim_time = True`, so its
   clock is the simulator's clock even without a params file.
2. **Subscribes** to `/scan` and `/ego_racecar/odom` with a **queue depth of 1,
   "best effort"**. The callbacks only store the newest message. They never
   compute anything, so the planner always reads the freshest data. (Earlier,
   a depth-10 queue meant we always read the oldest of 10 messages.)
3. **Runs a timer every 25 ms of simulated time** (`_control_step`). Each tick
   it:
   - updates the `LatchClock` (section 4.5) with a (simulated time, wall time)
     pair;
   - calls `core.step()` to get a command;
   - publishes it on `/drive`;
   - tells the core the simulated time at which the command will actually take
     effect (`note_published`).
4. **Publishes a debug marker and logs health.**
   - Every 10th tick (4 Hz) it publishes an arrow on `/driver/markers`. In RViz
     this is the **blue-green arrow**, 1.5 m long, starting at the rear axle and
     pointing in the direction of the *commanded steering angle*. It shows where
     the driver is trying to steer, not the path.
   - Every 5 s it logs:
     - `dt_measured`: the spacing between scans;
     - the steering-estimate residual and the learnt real-time factor;
     - how often the safety limits fired.
   - A control step that arrives late (more than 60 ms of simulated time after
     the last one) or takes longer than 20 ms of wall time is logged as a
     `slow step` line, together with how old the scan and pose were.

The odometry callback also extracts the **sideslip angle** β (the angle between
where the car points and where it actually moves): `atan2(linear.y, linear.x)`.
It extracts the **yaw rate** (how fast the car rotates) from `angular.z`. Both
come free in the message, and the dynamic model needs them.

### 4.2 Parameters (`driver_params.py` and `config/driver_params.yaml`)

- **One table** (`PARAM_SPECS`) lists every parameter with its default. It
  includes every field of `DriverConfig` automatically, so a field cannot exist
  without being declared.
- **The defaults in code are the race car.** The judges run a bare `ros2 run`,
  which never reads our YAML file. So the YAML is only for experiments, and
  `test_driver_params.py` enforces that every value in it equals the code
  default.
- At startup the node logs every value. The first line is `params_source =
  CODE_DEFAULTS` (code defaults are running) or `params_source =
  driver_params.yaml` (the file was loaded). That proves which configuration
  ran.
- **Validation** (`driver_params.validate`) refuses to start on a nonsense
  config, for example:
  - a control period different from the MPC step;
  - a block map or time schedule of the wrong length;
  - a non-positive acceleration;
  - a sector scale outside (0, 1.5];
  - an invalid warm-up or unstick setting.
- **Why this file exists:** our original YAML had the whole MPC section as a
  top-level key, and ROS reads a top-level key as a *node name*. So every MPC
  setting in it was silently ignored for days. One table with tests removed
  that whole class of bug.

### 4.3 The racing line (`raceline.py`)

The racing line is the path the car tries to follow. A good line uses the full
track width to make corners gentler: the gentler the corner, the faster you can
take it. Building it takes six stages.

**Stage 1: load the map.** The occupancy-grid image (a picture where dark
pixels are walls) is read with the map's own threshold settings. It is
read-only; we never modify `maps/`.

**Stage 2: seal the cone rows (rule 11).**

- Find every small, free-standing blob in the map: 0.1–0.9 m across, at most
  400 pixels. These are the cones.
- Draw a 3-pixel-thick barrier between every pair of cones closer than 1.1 m.
- Apply a **morphological closing** with a 0.3 m radius. This is an image
  operation that fills any gap narrower than 0.6 m between obstacles.

After this, no line can pass between cones. Real corridors are wider than
0.6 m, so they stay open.

**Stage 3: how far can the car move sideways?** At each point of the
centreline:

- Look along the **normal**, the line at right angles to the track direction,
  to the left and to the right.
- March outwards in 2.5 cm steps until the car's centre would be closer than
  **0.405 m** to a wall. That is half the car's width (0.155 m) plus our
  safety margin of 0.25 m.

This gives a left and a right limit at every point.

**Stage 4: the minimum-curvature line.** We move each point sideways by some
offset `a` (within its limits), to make the path as smooth as possible:

```
minimise   Σ |p[i-1] − 2·p[i] + p[i+1]|²      where p[i] = centre[i] + a[i]·normal[i]
subject to left_limit[i] ≤ a[i] ≤ right_limit[i]
```

`p[i-1] − 2p[i] + p[i+1]` (the "second difference") measures how sharply the
path bends at point i. The path is linear in the offsets, so this is a
**box-bounded linear least-squares problem**: the simplest kind of quadratic
optimisation (QP). SciPy's `lsq_linear` solves it in about half a second.

This is the method of Heilmeier et al. 2019, *"Minimum curvature trajectory
planning and control for an autonomous race car"* (Vehicle System Dynamics).

**Stage 5: re-linearise and re-weight.** Second differences only approximate
curvature when points are evenly spaced, so we repeat: resample, recompute the
normals and limits, solve again. We do this 6 times. Each round also **weights
up** points where the previous line was tighter than 0.6 /m (weight =
κ / 0.6). Plain minimum curvature spreads the bending evenly, which can leave
one apex sharper than the steering lock allows. The weights trade a little
length elsewhere for a gentler apex.

If the weighted solve fails, a plain 3-round solve is used, and the line is
flagged `reweighted = False`. The driver refuses that fallback line, because the
speed tuning does not fit it.

The solve runs at 0.2 m point spacing. The delivered line has 0.1 m spacing.

**Stage 6 (offline only, `--refine`): minimum lap time.** Minimum curvature is
only a stand-in for "fastest". So we refine it directly for lap time:

- **50 control points** around the lap. Each holds a sideways offset, and a
  smooth periodic spline joins them.
- The **cost** is the *ideal lap time* (drive the line at its speed profile,
  section 4.4), plus a large penalty if any point gets closer than 0.405 m to a
  wall.
- **L-BFGS-B** (a standard optimiser that respects box limits) minimises it,
  using finite-difference gradients (nudge each offset by 1 mm and see what
  happens to the lap time).
- 3 rounds, re-linearising the normals and limits around the new line each
  time.
- The result is rejected unless it keeps the clearance *and* actually lowers
  the ideal lap.

On the shipped settings this cut the ideal lap from **8.169 s to 8.009 s** and
took about 230 s. That is too slow for the 60 s startup budget, so the result
ships as data: `config/icra26_raceline.csv` (744 points, 0.1 m apart, 74.4 m).
Its header records `margin_m: 0.25` and `method: min-time`.

The profile the line is refined against (`REFINE_PROFILE`) is a **fixed
constant in the code**, not read from the driver's config. So the CSV is
reproducible from the map alone, even if the driver is retuned later. It must
be regenerated with `OMP_NUM_THREADS=1`. We measured that multi-threaded maths
libraries move the QP line by about 0.000001 m, and the lap-time search
amplifies that to about 9 cm.

**At startup, the driver chooses its line in this order**
(`DriverCore._raceline_reference`):

1. **The shipped min-time CSV**, but only if its header says `method:
   min-time` *and* every point is at least 0.405 m from the sealed walls of
   the map it is about to drive. Shipped data is trusted only as far as the map
   agrees with it.
2. Otherwise, **build the minimum-curvature line from the map** right there
   (about 3 s).
3. If that fails, the cached CSV (if its margin is at least 0.25 m and it did
   not fail the map check).
4. If that fails, the organisers' **centreline**.

If the car is not on the tuned min-time line, every raised sector scale
(section 4.4) is clamped to at most 1. On the centreline the whole lap gets one
uniform, cautious budget (27 × 0.85 = 22.95 m/s²). The "where to back off"
sectors are positions along *our* line, and would land on the wrong corners of
another line.

### 4.4 The speed profile (`speed_profile.py`)

The speed profile is the fastest speed the car should hold at every point of
the line. It is computed once at startup, and recomputed if the line changes.

**Step 1: the corner limit.** In a steady turn, sideways acceleration =
speed² × curvature. Keeping it within the slip budget gives:

```
v_corner = sqrt( a_lat_max × sector_scale / κ )
```

**Step 2: the steering-lock limit.** At speed, this car needs *more* steering
than the geometry alone suggests (it **understeers**):

```
wheel angle needed = wheelbase × κ + K × v² × κ        (K = 0.00279 rad per m/s²)
```

Setting that equal to 95% of the lock (`lock_margin`) gives the fastest speed
the wheel can hold the corner at. It only binds in the tightest corner. At
s ≈ 42 (radius 1.04 m) the lock allows **5.47 m/s** and the slip budget would
allow 5.58 m/s, so **the steering lock, not the slip budget, sets the speed
there.**

**Step 3: the top-speed cap**, `v_max` = 12 m/s. It binds only at the end of
the longest straight. The car's power limit tops out near 12.1 m/s anyway.

**Step 4: the braking pass** (backwards round the lap). Walking backwards from
each point, make sure the car can slow down in time:

```
v[i] ≤ sqrt( v[i+1]² + 2 × braking × distance )
```

**Step 5: the acceleration pass** (forwards). Make sure the car can actually
speed up that fast, including the power limit above 7.3 m/s.

Both passes go **twice round the lap** (the track is a loop), so the corner
just after the start line is braked for correctly.

**The g-g braking limit (`gg_brake`).** Because braking in a corner spins the
car (section 2.5), braking and cornering share one budget. The braking allowed
into a point shrinks as that point's corner uses more of its slip budget:

```
braking allowed = a_brake_max × max(gg_floor, 1 − r⁵)^(1/5)
r = (v² × κ) / budget         (the share of the corner budget in use, 0 to 1)
```

With `a_brake_max` = 9, `gg_floor` = 0.1 and exponent 5:

| Share of corner budget in use (r) | Braking allowed |
|---|---|
| 0 (straight) | 9.0 m/s² |
| 0.5 | 8.9 m/s² |
| 0.8 | 8.3 m/s² |
| 0.9 | 7.5 m/s² |
| 1.0 (at the corner limit) | 5.7 m/s² |

The name comes from the "g-g diagram" racing engineers use: forward
acceleration plotted against sideways acceleration. A driver does the same
thing: brake in a straight line, then turn. The exponent sets the shape:

- 2 is a true ellipse, which brakes very little once the car is cornering
  (2.8 m/s² at the limit).
- 5 is "squarer": it keeps near-full braking until close to the limit.

We started at 2 and moved to 5, once the planner itself handled braking safely
(section 7, Phase 7).

**Curvature smoothing (`curvature_smooth` = 21).** Curvature from three
neighbouring points is noisy at 0.1 m spacing, so it is averaged over 21 points
(about 2 m). Wider smoothing rounds off the curvature peaks, which raises corner
speeds. On the earlier line, 5 points gave an ideal lap of 9.201 s and 13
points gave 8.809 s. 21 was pinned together with `a_brake_max` 9 and exponent 5
after harness runs (the stronger 25 / 9.5 / 6 combination crashed at s 54).
This smoothing is for the profile only. The MPC uses its own.

**Sectors (`sector_scale`).** These scale the slip budget on stretches of the
lap, given as triples (start s, end s, scale). Below 1 means "be gentler here";
above 1 (up to 1.5) means "this corner can take more". Where sectors overlap,
the lower scale wins. Our values:

| Sector (s, m) | Scale | Budget | Why |
|---|---|---|---|
| 3.1–7.1 | 0.95 | 25.7 m/s² | First corner. It collided at s ≈ 4.8–5.2 on the new line without this. |
| 8.1–21.3 | 1.20 | 32.4 m/s² | Raised one step below where it crashed (1.35 hit). |
| 28.3–43.2 | 1.11 | 30.0 m/s² | Hairpin complex. 1.3 crashed in the harness. 1.2 was clean in the harness but **hit s 30–36 in 3 of 10 judged runs**, so it was cut to 1.11. |
| 43.2–47.2 | 0.85 | 23.0 m/s² | Chicane exit. A tracking wobble collided at s ≈ 45.3 without it. |
| 49.2–57.2 | 1.11 | 30.0 m/s² | 1.15 hit at s ≈ 54. |
| 57.2–74.4 | 1.20 | 32.4 m/s² | Fast final section. |
| (21.3–28.3, 47.2–49.2) | 1.0 | 27 m/s² | Raising 21–28 above 1.0 hit at s 23.2. |

Each raised corner sits **one notch below the value that collided** in our
offline harness ("one step below the cliff"). This is the most hand-tuned part
of the car; see section 11.2 for how we explain it.

**Result on the shipped line:**
- The race profile runs from 5.47 to 12.0 m/s, with an **ideal lap of
  8.056 s**.
- Without any sectors it would be 8.291 s.
- The warm-up profile runs from 3.1 to 12 m/s, with an ideal of 12.6 s.

The driven lap (judged 8.04 s, harness 7.97 s) comes in slightly under the
"ideal". The ideal is a guide computed along the line's points, not an exact
floor: the car does not drive exactly on the line or exactly at the profile
speed.

### 4.5 Dead time: the latch clock, steering estimator and latch predictor

Three pieces work together to answer one question: **where will the car be
when the command we are computing now actually takes effect?** "Latch" is our
word for the moment the simulator picks up a new command and starts acting on
it.

**`LatchClock` (in `driver_core.py`, used by the node only).**
- The node's simulated clock stands still while our code runs, but the
  simulator keeps going. So a command published after a 5 ms solve lands about
  `5 ms × real-time factor` of simulated time later.
- `LatchClock` learns the real-time factor (RTF) from how far simulated time
  and wall time move between control steps, smoothed by 5% per step.
- **The launch-crash fix:** at startup, the node's first clock update can jump
  0.5–1 s of simulated time in about 3 ms of wall time. That looks like RTF 340.
  - The first, unsmoothed sample made every latch time about 0.6 s late during
    the launch.
  - The steering estimate ended up 0.9–1.1 rad wrong, and the car steered into
    the first wall.
  - It happened in 3 of 3 judged runs whose log started with that jump, and in
    0 of 17 others.
  - Fix: any sample implying more than 10× real time is ignored
    (`LATCH_RTF_MAX = 10`). Judged RTF is 1.8–2.4, with the highest spike seen
    at 6.85.

**`SteeringEstimator` (in `driver_core.py`).** Nothing measures the wheel
angle, so we predict it:

- We know every steering command we published, and the simulated time it
  latched.
- We **replay the simulator's own servo formula** (section 2.2), one 10 ms step
  at a time, to know where the wheel is at the moment the pose describes.
- To stop slow drift, we gently pull this estimate towards the angle implied by
  the measured yaw rate, `atan(wheelbase × yaw rate / (v cos β))`. The pull has
  a 0.5 s time constant, and is used only above 0.8 m/s, where that formula is
  reliable. The replay is primary; a stronger pull would feed yaw-rate noise
  into the planner.
- If simulated time jumps backwards by more than 0.1 s, the simulator was
  reset. The estimate is re-centred and the old commands are forgotten.

**`LatchPredictor` (`latch_predictor.py`).** The full version of the same idea.
From the measured state (position, heading, speed, slip, wheel angle, yaw rate)
at the pose's timestamp, it replays every command we published since then
through the simulator's own equations:

- the servo (`pid_steer`);
- the speed controller (`pid_accl`);
- the single-track dynamics;

all one 10 ms physics step at a time. It runs up to the predicted latch time of
the next command: the last publish time plus one control period (or longer, if
solves have been overrunning the timer). It returns the predicted state and the
distance travelled.

**The MPC then plans from that predicted state**, and the speed limits act from
the predicted speed. In effect, the planner sees the car where it will be, not
where it was about 0.1 s ago.

### 4.6 The MPC (`mpc_controller.py` + `vehicle_model.py`)

#### What it decides

Two numbers per move: a **steering command** (within ±0.4189 rad) and an
**acceleration** (within −9.0 and +9.51 m/s²).

#### What it predicts with: the state and the model

The **state**, the numbers that describe the car at one instant, has 7 values:
x, y, heading, speed, sideslip β, **wheel angle δ**, and yaw rate r.

The **model** (`VehicleModel` with `model_type = 'ST'`) is the judge's own
single-track equations, rewritten in three ways:

- **Positions are at the rear axle**, because odometry and the racing line are
  at the rear axle. The simulator integrates the centre of mass.
- **The slip and yaw-rate equations are stepped exactly** (`_lateral_zoh`).
  - At low speed those equations are very "stiff": the simple Euler method
    (step = rate × time) becomes unstable below about 2.5 m/s on a 25 ms step.
  - For a held speed they are linear, so we use the exact solution, a 2×2
    matrix exponential written out by hand.
  - Heading and position then follow by the trapezoid rule.
  - Below 0.5 m/s the simulator itself switches to simple (no-slip) geometry,
    and so do we.
- **The servo is part of the model.** The command only moves the wheel, through
  a servo model; the car turns with the wheel as it actually stands. So every
  predicted path respects the 0.08 rad per 25 ms rate limit, at no cost in
  extra variables or constraints.

The servo model is a smooth version of the real one:

```
move = d_max × x / (1 + |x|⁸)^(1/8),     x = 0.6903 × error / d_max,     d_max = 0.0801 rad per 25 ms
```

- For small errors it closes 69% of the gap per 25 ms; for big ones it moves
  0.0801 rad.
- The two constants were fitted by least squares against the real plant (a
  test, `test_steering_actuator.py`, pins the fit).
- It is smooth on purpose. The optimiser estimates slopes by nudging inputs
  (finite differences), and a hard clip has zero slope once saturated, so the
  optimiser would go blind.

We also have a simpler **kinematic model** ('KS': no sideslip). We measured
both against the real plant with `model_error.py`. Above a few m/s the
kinematic model drifts, because the car really does slide sideways, so the race
car uses ST.

#### How far ahead it looks: horizon and time schedule

- **8 prediction steps** whose lengths grow: 25, 25, 50, 50, 75, 75, 100,
  100 ms (`dt_schedule`). That covers **0.5 s**, enough to see the next
  corner.
- The first step must equal the 25 ms control period: the command is held
  exactly that long.
- Why not just more steps? Solve time grows faster than the number of steps.
  Before move blocking (one free move per step) we measured 30 ms at 8 steps,
  139 ms at 12 and 239 ms at 16. Stretching the later steps buys look-ahead
  for free.

#### How many choices it makes: move blocking

The 8 steps share only **3 distinct moves**: `block_map = [0, 1, 1, 2, 2, 2,
2, 2]`. Move 0 covers step 1, move 1 covers steps 2–3, and move 2 covers steps
4–8. So the optimiser chooses only **6 numbers** (3 moves × steering and
acceleration). This is called **move blocking**, and it is the main reason the
solve fits in time.

#### What "good" means: the cost function

For each predicted state, compared with the reference (the racing line, see
"Where the target is" below):

| Term | Weight | Meaning |
|---|---|---|
| position error (x and y) | `q_pos` 2.0 | Stay on the line |
| heading error | `q_yaw` 2.0 | Point (travel) along the line |
| speed error | `q_vel` 0.5 | Hold the profile speed |
| steering command² | `r_steer` 0.08 | Do not steer more than needed |
| acceleration² | `r_accel` 0.01 | Mild preference for gentle throttle |
| change in steering² | `r_delta_steer` **3.0** | **No chattering**, including the jump from the last command actually sent |
| change in acceleration² | `r_delta_accel` 0.005 | Smooth throttle |

The heading error compares the **direction of travel** of the rear axle with
the line's direction, not the direction the car points. A sliding car's body
points into the corner while it moves along the line. Charging that as an error
would fight the slip the corner needs.

`r_delta_steer` = 3.0 was found by a sweep (section 7, Phase 1). Our original
weights made a 0.2 m jump in the target about **57× cheaper** than the steering
reversal it caused, so the optimiser chose to chatter.

The MPC has **no wall constraints**. It does not know where the walls are. It
follows a line that already keeps 0.405 m from every wall, and the LiDAR
emergency cap (section 4.7) catches what is left. This keeps the optimisation
small and fast.

#### Where the target is: the reference window

At each solve the MPC needs "where should the car be at each future step".

- **Locating the car on the line.** It finds the nearest line point, but only
  near the last one: the track doubles back, so a point across the infield can
  be closer in a straight line. It then **projects the car onto the line
  segment**, to get a continuous arc position s. That position keeps counting
  past the lap length, so it never jumps at the finish line.
  - An earlier version snapped s to the nearest waypoint, 0.2 m apart. The
    target then jumped by up to 0.2 m between solves, and that was the main
    cause of the steering chatter.
- **Placing the future targets at a reachable pace** (`_reachable_arc`). They
  are spaced as if the car chases the profile at its real limits: speeding up
  by at most the car's acceleration ceiling, and slowing by at most 9 m/s².
  - Before this fix (commit `525ce12`), the targets moved at the car's
    *current* speed. Whenever the car tried to speed up, the MPC saw itself
    running ahead of the target and pushed back.
  - That hidden accelerator limit left a 0.41 s gap between the planned and
    the driven lap; the fix closed it to about 0.08 s.
- **Wrapping round the lap.** The window wraps across the finish line.
  Originally it did not: the spawn sits near the end of the centreline array,
  so the targets piled up on the last point behind the car. The car's best move
  was then to stop. This caused several "stuck" runs at the end of September.

#### How it solves

- **SLSQP** (Sequential Least-Squares Programming, from SciPy) handles the 6
  bounded variables, capped at 25 iterations.
- **Warm start:** it begins from the last plan, shifted one control period
  forward and refitted to the blocks. So most solves only need a few
  iterations. With the iteration cap, each solve continues from the previous
  one rather than starting from scratch.
- **A "fused" objective:** the cost and its finite-difference gradient are
  computed in plain Python floats with no per-step arrays. Each nudge of a move
  only re-simulates from that move's first step onwards. This gives the same
  maths as SciPy's own differencing, but much faster.
- **A hard 15 ms wall-clock budget.** If the solve runs past 15 ms, it is
  abandoned. The car keeps following the **previous plan, advanced one step**,
  so the command is continuous with the last one rather than random. The same
  happens if the solver errors or returns NaN.
- Typical solve: about 4–5 ms of wall time on our laptop (measured in judged
  runs of the sub-10 car), well inside the budget.

#### Who brakes: the MPC owns the braking (`mpc_full_brake`)

Originally the MPC's lowest allowed acceleration was −0.3 × 9.51 =
**−2.85 m/s²** (a conservative default from the template). Harder braking came
from the speed-profile cap overriding the MPC (section 4.7), which ignores the
car's dynamics.

When we instrumented 63 crashes at budgets 23–27, all 63 were **braking yaw**.
The MPC sat at its −2.85 limit while the profile override braked at −5 to
−7.8 m/s², a deceleration (and rear unloading) the MPC's model never saw coming.

Now the MPC may plan braking down to −9 m/s² itself. Its model includes the
load transfer, so it brakes in a way it can predict. The profile override is
limited to 3 m/s² of extra braking (`profile_brake_cap`).

### 4.7 Safety layers and output conditioning (`DriverCore._condition`, `longitudinal.py`)

Everything the MPC wants passes through one function before it reaches the
car. In order:

1. **Bad numbers → coast.** If the steering, acceleration or any speed limit is
   NaN or infinite, the car **coasts**:
   - the wheel unwinds towards straight at 1 rad/s;
   - speed is shed at 7 m/s².
   It never snaps the wheel to zero.
2. **Steering slew limit.** The wheel may move at most `3.2 × 0.025 =
   0.08 rad` from the last command. That is exactly what the servo can follow,
   so this removes only motion the wheel could never have made, and adds no
   lag. A clean run should almost never hit this limit.
3. **Acceleration clamp:** between −9.0 (`a_brake_max`) and +9.51
   (`accel_limit`) m/s².
4. **Profile cap.** The speed at the end of this 25 ms hold must not exceed
   the speed profile there. But this cap can force **at most 3 m/s²** of
   braking by itself (`profile_brake_cap`). Harder braking must come from the
   MPC (section 4.6).
5. **Emergency cap (LiDAR).** Detailed below. It may brake as hard as the car
   can (9.51 m/s²), past every other limit, because it is the safety net.
6. **Speed-loop inversion.** The final acceleration is turned into the exact
   `/drive` speed that makes the simulator's controller deliver it over the
   next 25 ms, starting from the *predicted* speed at the latch (section 2.3).
   The speed sent is never below 0: the car only reverses in unstick.

**The emergency cap in detail** (`_path_cap`):

- It takes the MPC's own predicted path (0.5 s) from the predicted latch
  position, and imagines a **corridor one car wide** (±0.155 m) along it.
- If the car is following the line (within 0.5 m of it, heading within
  0.3 rad), the corridor is **continued along the racing line** beyond the end
  of the plan, out to the full stopping distance plus 1 m.
  - Without this, the corridor carried straight on past the plan's end, into
    the *outside* wall of the next corner. The car then braked at up to
    10.6 m/s² where the profile asked for 2.
  - We found every single emergency-cap trigger at budget 22 was one of these
    false alarms (commit `42cb78f`).
- The LiDAR points are placed in the map from the measured pose.
- **Clearance** = how far along the corridor the first wall points are. It is
  the 10th-percentile distance, not the mean, so one long beam down a gap
  cannot hide a near wall. With few beams, that is the minimum: it fails
  towards braking.
- Allowed speed = `sqrt(2 × 7 m/s² × (clearance − bumper distance − distance
  covered during this hold))`, between 1 m/s and 12 m/s. If nothing is within
  stopping distance, there is no cap.
- If there is no usable plan, a simpler version looks along a straight
  corridor towards the line point 1–2 m ahead (`_speed_cap`).

On a clean lap the emergency cap essentially never triggers. It is the net
under the trapeze, not part of the act.

### 4.8 Scan preprocessing

`preprocess()` turns the raw LiDAR array into clean numbers:
- NaN → 0 (treated as an obstacle right at the sensor, which fails safe);
- infinity → 8 m;
- everything clipped to 0–8 m (`max_range`).

### 4.9 The reactive fallback (follow-the-gap)

`DriverCore.plan` always computes a simple follow-the-gap steering:

1. Look at ±100° in front.
2. Blank out a "bubble" around the nearest obstacle.
3. Find the widest run of beams longer than 1.5 m.
4. Steer at 85% of the angle to its middle.

It is used **only if there is no pose or no reference line** (for example, the
map files are missing). Then the car drives at 3 m/s on LiDAR alone. In a
normal run it is computed but not used.

### 4.10 Warm-up mode

The out lap and warm-up lap are not timed, but their collisions count towards
disqualification.

- At the sub-9 tune, warm-up laps had wall contacts in 2 of 13 judged runs
  (3 hits in total), while 130 timed laps had none.
- Later warm-up contacts were near-misses at the chicane exit (s 40–50), early
  in a run, when the host was busiest.
- The harness showed the warm-up lap had the **thinnest** wall clearance of any
  lap there: 0.28 m.

So until the car has crossed the start line once (ending the out lap) **and**
then reached s = 56 m of the warm-up lap:

- the profile uses **0.35 of the slip budget** from s 0 to 56 (about
  9.5 m/s²);
- every acceleration limit is **halved**: the profile's, the MPC's upper bound
  and the conditioner's.

After s 56 the car switches to the race profile, and is back at full speed well
before the line, so timed lap 1 starts at race pace.

- The warm-up lap takes about 12.5 s instead of 8.
- In the harness, the closest approach to a wall on the warm-up lap went from
  0.28 m to 0.33 m.
- Timed laps are unchanged.

The switch keeps the MPC's tracking state. Only the speed targets and limits
are swapped.

Counting crossings uses the MPC's arc position, reduced into one lap. A jump of
more than half a lap counts as a crossing. A backwards crossing (jitter on the
line) must be undone by a forward one before it counts, the same rule the
simulator uses.

### 4.11 Unstick: getting off a wall

Once in a judged run, after one hit, the car sat against the wall at full lock
and full throttle for 4 s. The referee counted a collision every second and
disqualified it. Nothing in the driver ever asked for reverse.

Now (`DriverCore._unstick`):

- **Pinned** means: slower than 1 m/s while the last command asked to move, for
  0.5 s.
  - The 1 m/s threshold came from judged runs: a car scraping a wall still
    creeps at 0.37–0.5 m/s, so an older 0.3 m/s threshold never fired.
  - When racing, the car is never under 4 m/s.
- It is **armed only after the car has once driven faster than 2 m/s**. Before
  the simulator applies the first commands, the car sits still on the grid
  while asking to move. That looked like "pinned", and once reversed the car
  off the grid.
- When pinned, it **reverses at 1 m/s for 0.6 s with the wheels turned to full
  lock the opposite way** to the last forward steer (counter-steer), like the
  first move of a three-point turn. If the last steer was nearly straight
  (under 0.05 rad), it reverses straight. That swings the nose away from the wall,
  towards where the driver wanted to go. Reversing straight left the nose at
  the wall, and full lock put it straight back on (one judged run: 15 reverses,
  disqualified).
- If it gets pinned again within 3 s, each reverse is longer: 0.6, 0.9, 1.2,
  then at most 1.5 s.
- While reversing, the MPC is not run.

### 4.12 Simulator resets

If simulated time jumps backwards (someone reset the car with `/initialpose`),
everything that remembers the past is cleared:
- the steering estimate, latch history and slew memory;
- the MPC's tracking and plan;
- the warm-up (it starts again from the grid);
- unstick.

---

## 5. One control cycle, step by step

This is exactly what happens every 25 ms of simulated time, in order, with the
function names. (`driver.py` → `DriverCore.step`.)

1. **Timer fires** (`Driver._control_step`). Read the simulated clock and the
   wall clock; feed the pair to `LatchClock`.
2. **No scan yet?** Coast (car on the grid before the first scan).
3. **Same scan as last time?** (The timer can beat the next scan.) Re-send the
   last command instead of solving the same data twice.
4. **Wheel estimate** (`SteeringEstimator.predict_to`, then `.correct`): replay
   the servo up to the pose's timestamp, then nudge the estimate towards the
   yaw-rate-implied angle.
5. **Reset check.** If the simulator was reset, clear all memory (section
   4.12).
6. **Predict the latch state** (`_predict_latch` → `LatchPredictor.predict`):
   where the car will be when this command lands.
7. **Unstick check** (`_unstick`). If pinned, return a reverse command and
   stop here.
8. **Clean the scan** (`preprocess`).
9. **Plan** (`plan`):
   - compute the follow-the-gap fallback steering;
   - if there is a pose and a line, run the MPC from the latch state
     (`get_control`). Record its arc position (for warm-up counting) and its
     predicted path;
   - work out the emergency cap (`_path_cap`) and the profile speed where this
     hold ends (`_profile_speed`).
   If the MPC throws an error, return "no plan" (coast).
10. **Condition** (`_condition`): slew limit, acceleration clamp, profile cap,
    emergency cap, speed-loop inversion → a `DriveCommand(steering, speed)`.
11. **Publish** on `/drive`.
12. **Record** (`note_published`): the command and its predicted latch time
    (clock reading + solve wall time × learnt RTF). This feeds the steering
    replay and the latch predictor for the next cycles. The interval between
    publishes is smoothed, so overrunning solves are accounted for.
13. Every 10th cycle, publish the RViz arrow. Every 5 s, log the health
    statistics. Log any slow step.

---

## 6. Every tuned number, and why it has that value

All of these are `DriverConfig` fields in `driver_core.py` (or MPC parameters in
`driver_params.py`), mirrored in `config/driver_params.yaml`. "Old car" is the
value before the latest change, which is what the switch reproduces.

### 6.1 Speed and profile

| Parameter | Value | Meaning and evidence |
|---|---|---|
| `a_lat_max` | 27.0 m/s² | The slip budget. 27 was clean on the full harness grid; 28 hit the chicane exit (s 45.4) in 1 of 3 cells; 30 hit everywhere. Old car: 21. |
| `sector_scale` | see 4.4 | Per-stretch budget multipliers, each one notch below its crash value. |
| `v_max` | 12.0 m/s | The top speed. An ideal-lap estimate put it at 0.14 s/lap faster than 10 m/s. The power limit tops out near 12.1. |
| `a_accel_max` | 9.51 m/s² | The acceleration the profile plans: the car's own maximum. |
| `a_brake_max` | 9.0 m/s² | The braking the profile plans, and the most the MPC may use. 9.5 (with exponent 6 and smoothing 25) crashed at s 54. Old car: 6. |
| `lock_margin` | 0.95 | The profile uses at most 95% of the steering lock. |
| `gg_brake` | true | The braking/cornering budget is shared (4.4). Off = plain profile, exactly. |
| `gg_floor` | 0.1 | At the corner limit, braking is still 0.1^(1/5) = 63% of the maximum. |
| `gg_exponent` | 5.0 | The shape of the ellipse (2 = true ellipse, old car). |
| `curvature_smooth` | 21 points | The profile's curvature averaging (old car 9). |

### 6.2 MPC braking ownership

| Parameter | Value | Meaning and evidence |
|---|---|---|
| `mpc_full_brake` | true | The MPC may brake to −9 m/s², not −2.85. |
| `profile_brake_cap` | 3.0 m/s² | The most braking the profile cap may force by itself. 6 still spun at budget 23; 0 lost the chicane exit at 27; 3 was clean to 27. Old car: 9.51 (no cap). |
| `beta_limit` | 1.0 rad | A measured slip bigger than this is passed to the MPC as 0, because it means reversing or a spin, not cornering. 0.5 (old car) was too small from budget 25: the car really slides past 0.5 rad in some corners, the MPC lost its slip state, and spun at s ≈ 16. |

### 6.3 Line

| Parameter | Value | Meaning and evidence |
|---|---|---|
| `reference` | "raceline" | Track the racing line ("centerline" is the alternative). |
| `raceline_margin` | 0.25 m | Extra clearance beyond the car's half-width. 0.10 was too thin at speed (disqualified). 0.20 brought back a collision at s ≈ 65.5. |

### 6.4 Safety, warm-up and unstick

| Parameter | Value | Meaning and evidence |
|---|---|---|
| `a_emergency` | 7.0 m/s² | The braking the emergency cap assumes when computing a safe speed; also how fast a coast sheds speed. |
| `cap_follows_reference` | true | Continue the emergency corridor along the line (removes the false alarms). |
| `crawl_speed` | 1.0 m/s | The emergency cap never asks for less than this. |
| `max_range` | 8.0 m | LiDAR readings are clipped here. |
| `stop_margin` | 0.0 m | Extra stopping room beyond the bumper (none needed). |
| `steer_limit` | 0.4189 rad | The real steering lock. |
| `coast_steer_rate` | 1.0 rad/s | How fast the wheel unwinds when coasting. |
| `accel_limit` | 9.51 m/s² | The most acceleration passed to the car. It must match what the MPC's target window assumes, or the target outruns the car. |
| `predict_latch` | true | Plan from the predicted latch state (false = from the old measured state). |
| `warmup_crossings` | 1 | Warm-up mode until 1 start-line crossing… |
| `warmup_end_s` | 56.0 m | …and then s 56 of the next lap. |
| `warmup_sector_scale` | (0, 56, 0.35) | The warm-up budget. |
| `warmup_accel_scale` | 0.5 | Halve all acceleration in warm-up. |
| `unstick_after` | 0.5 s | Pinned this long → reverse (0 = off). |
| `unstick_reverse_s` | 0.6 s | The first reverse length (grows on repeats, to 1.5 s at most). |
| `unstick_speed` | 1.0 m/s | Reverse speed. |

### 6.5 MPC

| Parameter | Value | Meaning and evidence |
|---|---|---|
| `control_period` = `mpc.dt` | 0.025 s | 40 Hz. The node refuses to start if these differ. |
| `vehicle.model` | "ST" | The dynamic single-track model. |
| `mpc.prediction_horizon` | 8 | Prediction steps. |
| `mpc.dt_schedule` | 25, 25, 50, 50, 75, 75, 100, 100 ms | 0.5 s of look-ahead. |
| `mpc.block_map` | [0,1,1,2,2,2,2,2] | 3 free moves = 6 variables. |
| `mpc.control_horizon` | 3 | Must equal the number of blocks. |
| `mpc.maxiter` | 25 | SLSQP iteration cap. |
| `mpc.solve_budget_ms` | 15 ms | Wall-clock cap; past it, follow the previous plan. |
| `mpc.q_pos / q_yaw / q_vel` | 2.0 / 2.0 / 0.5 | Tracking weights. q_pos turned out insensitive in sweeps. q_yaw 4 was faster but chattered. |
| `mpc.r_steer / r_accel` | 0.08 / 0.01 | Effort weights. |
| `mpc.r_delta_steer` | 3.0 | The anti-chatter weight. The sweep of 0.1–8 found 3.0 to be the knee (the first value that passed the smoothness gate). Lower chatters (0.1: +0.7 s/lap). |
| `mpc.r_delta_accel` | 0.005 | Throttle smoothness. |
| `mpc.feedforward` | false | Seed the first steering move from the curvature (an option, off). |
| `servo.alpha / servo.d_max` | 0.6903 / 0.0801 rad | The servo model, fitted to the plant. |

### 6.6 Constants in code (not parameters)

| Constant | Value | Why |
|---|---|---|
| `LATCH_RTF_MAX` | 10 | Clock jumps are ignored (4.5). |
| `UNSTICK_SPEED` / `UNSTICK_ARM_SPEED` | 1.0 / 2.0 m/s | 4.11. |
| `UNSTICK_REPEAT_S` / `UNSTICK_REVERSE_MAX_S` | 3 s / 1.5 s | 4.11. |
| `CAP_CONTINUATION_MAX_OFFSET` / `_HEADING` | 0.5 m / 0.3 rad | When the car counts as "following the line" for the corridor continuation. |
| `PATH_CAP_SLACK` | 1.0 m | Extra corridor length beyond the stopping distance. |
| `STEER_CORRECT_TAU` | 0.5 s | The steering-estimate leash. |
| `REACTIVE_SPEED` | 3 m/s | The follow-the-gap fallback speed. |
| Plant constants (servo gain, rate, lock, mass…) | — | Deliberately **not** parameters: they are fixed by the simulator, and exposing them would only let our model drift from it. |

---

## 7. The journey: from a car that crashed to 8.04 s

Times are judged best laps from our own `evaluate.sh` runs (result files kept
on our machine in `results/`; only `results/submitted/` is committed).
"Harness" means our offline simulator (section 8).

| Date | Change | Judged best lap | Collisions |
|---|---|---|---|
| 17 Sep | The organisers' template ships no driver | — | — |
| 26–28 Sep | Follow-the-gap (reactive, LiDAR only) | ~28–30 s | 0 |
| 29–30 Sep | Pure Pursuit (built in parallel) | 17.83 s | 0 |
| 29 Sep | First MPC, tracking the centreline | 51.66 s | **7** |
| 30 Sep–1 Oct | **Phase 1:** faithful offline simulator, servo model, timing fixes, weight sweep | ~50 s | 0 |
| 2 Oct | **Phase 2:** racing line, speed profile, ST model, output unlocked, latch predictor | ~18.0 s | 0 |
| 2 Oct | Budget sweeps; back off only where it crashed | 14.2 s | 0 |
| 2–3 Oct | Line reworked for the tightest corner | 12.98 s | 0 |
| 5 Oct | **Phase 3:** MPC allowed to plan acceleration ("reachable pace") | 12.61 s | 0 |
| 5 Oct | Re-tune (budget 18 + sectors) | 11.08 s | 0 |
| 6 Oct | **Phase 4:** offline minimum-time line | 10.87 s | 0 |
| 8–9 Oct | **Phase 5:** g-g braking, smarter emergency cap, re-refined line | 9.85–9.87 s | 0 |
| 17 Oct | **Phase 6:** the MPC owns the braking; budget 27, top speed 12 | 8.82 s | 0 in 130 timed laps |
| 17–18 Oct | **Phase 7:** profile shape, re-refined line, raised corners | 8.03–8.05 s | (see below) |
| 18 Oct | **Phase 8:** clock fix, slow warm-up, counter-steer unstick | **8.038–8.047 s** | **0 in 10/10 runs** |

The collisions column describes the runs that closed each phase. The
experiments in between had crashes and disqualifications too. Every result
file is kept on our machine in `results/` (about 350 of them). Only the
submitted runs in `results/submitted/` are committed, as the rules ask.

### Phase 0: the baselines (follow-the-gap ~28–30 s, Pure Pursuit 17.83 s)

The team first built two standard drivers from the organisers' algorithm list:

- **Follow-the-gap** (section 4.9). It needs no map and no planning, so it was
  the quickest way round. It completed clean laps at about 28–30 s.
- **Pure Pursuit**, which steers towards a point a fixed distance ahead on a
  line. It reached **17.83 s**. We also tried it on a precomputed racing line.

Both kept improving more and more slowly. They react to where the car is *now*,
but what limits this car is what happens *next*: a command acts about 0.1 s
late, the wheel takes up to 262 ms to turn, and the car slides in corners. An
MPC can plan for all three. So we switched, knowing it would start slower.

The first MPC (51.66 s) was far slower than Pure Pursuit. It took until 2 Oct
(18.0 s) to match it, and every gain after that (down to 8.04 s) was beyond
what the baselines had reached.

### Phase 1: making the MPC drive at all (51.7 s with 7 crashes → 50 s clean)

The first MPC (29 Sep) finished its judged laps but hit walls 7 times. When we
reproduced it offline, the steering flipped direction 300 times in 700 steps,
slamming between full left and full right lock. Tuning weights did nothing, because
**our offline test harness was lying to us**: it tested the MPC against its own
perfect model, which had no servo and no delay.

We fixed it in nine steps (commits `254a9d1`…`d1cc2cf`):

| Step | Root cause, simply | Fix |
|---|---|---|
| 1–2 | The harness used our model, not the judge's | Rebuilt it to import the judge's physics, LiDAR and collision code, with the judge's timing (`plant_sim.py`, `scan_sim.py`, `lap_sim.py`). It then reproduced the judged failure. |
| (plan) | The target on the line was snapped to 0.2 m waypoints and jumped between solves; steering changes were ~57× too cheap in the cost | Continuous arc-length tracking; rebalanced weights |
| 3 | Dead time: a depth-10 queue meant the planner read data 10 messages old, and planning ran from the scan callback at an uneven rate | A fixed 25 ms timer and depth-1 queues |
| 4 | The MPC's "last command" memory was fed the *gap-follower's* steering, which jumps around | The MPC keeps its own command memory |
| 5 | Solves took 30–40 ms against a 25 ms budget | Move blocking (6 variables), an iteration cap and a 15 ms budget |
| 6 | The MPC thought the wheel moved instantly (it needs 262 ms lock to lock) | The servo became a state of the model; the wheel angle is estimated by replaying the servo |
| 7 | Raw optimiser output went straight to the car, and the wheel snapped to centre on any failure | Output conditioning and smooth coasting |
| 8 | Our tuning file was silently ignored (the ROS naming mistake, 4.2) | One parameter table, with tests |
| 9 | — | Re-swept the weights with the faithful harness: `r_delta_steer` 0.8 → 3.0 |

At the end of Phase 1 the car was stable and clean, but slow: about 50 s,
because every limit was still at a "safe" 0.8 m/s².

### Phase 2: from 50 s to 13 s by going faster on purpose (2–3 Oct)

- **A racing line** (minimum curvature, section 4.3) and **a proper speed
  profile** (section 4.4) replaced the centreline and its very cautious
  0.8 m/s² profile.
- **The dynamic ST model.** At speed the car slides sideways, and the kinematic
  model assumes it cannot. `model_error.py` measured the difference.
- **Look-ahead for free:** the stretched 25–100 ms step schedule (0.5 s).
- **Unlocked the output path** (`f5f0c8f`):
  - the speed-loop inversion (section 2.3); before it, the car got about 12% of
    the acceleration asked for;
  - the latch predictor (section 4.5);
  - the LiDAR caps along the path (section 4.7).
  Result: about 18 s.
- **Sweep, then back off locally.** We raised the budget, ran many laps,
  recorded *where* each crash happened (its arc position s), and lowered the
  budget only on those stretches. Result: 14.2 s.
- **Reworked the line** for the tightest corner: the curvature re-weighting in
  section 4.3, stage 5. Result: 12.98 s.

### Phase 3: the hidden accelerator limit (5 Oct, 12.98 → 11.08 s)

The MPC's targets moved at the car's *current* speed, so any attempt to speed
up looked like "running ahead of the target", and the position cost (4× the
speed cost) pushed back. With the plant allowing 9.51 m/s², the MPC asked for
a median of only 42% of it.

The **reachable-pace window** (`525ce12`) closed most of the gap between the
planned and the driven lap. A re-tune followed (budget 18 + 4 sectors,
`7489614`): judged 11.08 s.

This phase taught us an important testing lesson. A setting clean with fixed
solve times was *not* clean with real, varying solve times: a collision at
s ≈ 66 appeared only with jitter. From then on, collisions were screened with
measured timing (section 8).

### Phase 4: the minimum-time line (6 Oct, 10.87 s)

Offline lap-time refinement of the line (section 4.3, stage 6; `c15552a`). The
new line moved the crash sites, so the sectors were re-tuned. Judged 10.87 s,
with the harness about 0.05 s faster than the judged result.

### Phase 5: under 10 s (8–9 Oct, 9.85–9.87 s)

Above a budget of about 18 the car spun in several places. We traced the cause
(`47eefc7`):

- **The plant's yaw goes unstable under braking** (section 2.5). The old
  profile braked at a full 6 m/s² deep into corners, and the profile cap forced
  that on top of the MPC (which asked only about −2.9 there), so the rear
  stepped out.
- **Fix 1: the g-g braking limit** (section 4.4). This took the harness to
  9.80 s clean at budget 22.
- **Fix 2: the emergency cap's false alarms** (`42cb78f`). Its corridor ran
  straight past the plan into the outside wall of the next corner. It now
  follows the line (section 4.7).
- **Re-refined the line** at the new profile and pinned budget 21 with two
  small sectors (`186026c`).

We chose budget **21 over the faster 22** (9.71 s harness) on purpose: under
heavy host load, 22 hit in 2 of 18 runs, and 21 in 0 of 24. One hit costs
10 s; 0.15 s was not worth it. Judged: 9.85–9.87 s, many runs clean. These were
the first results we submitted.

### Phase 6: the MPC owns the braking (17 Oct, 8.82 s)

Every budget above 22 still crashed. We instrumented the first crash of 63
harness runs at budgets 23–27: **63 of 63 were braking yaw**. On corner entry
the MPC sat at its −2.85 m/s² floor, while the profile override added a
"blind" −5 to −7.8 m/s². The car was slightly over the profile because of lag,
so the override fired. The tracking itself was nearly perfect (9.816 s driven
against a 9.848 s ideal), so the time had to come from braking, not tracking.

Three changes together (`2fd33e5`):

1. **`mpc_full_brake`:** the MPC may brake to −a_brake_max itself.
2. **`profile_brake_cap` = 3:** the blind override may add at most 3 m/s².
3. **`beta_limit` = 1.0:** the faster car really slides past 0.5 rad, and
   hiding that from the MPC caused a spin at s ≈ 16.

Then budget 27 and top speed 12 m/s.
- Harness: 8.79 s, clean on the full grid, under 12-job host load and on the
  centreline.
- Judged: **8.815–8.825 s in 13 runs, 0 collisions in 130 timed laps.**
- But 2 of the 13 runs had a contact on the **warm-up** lap, which led to
  Phase 8.

We also tried a yaw-rate tracking cost in the MPC. It crashed the out lap at
every weight we tried, so we dropped it.

### Phase 7: profile shape and raised corners (17–18 Oct, 8.03–8.05 s)

With the MPC braking safely we could brake harder and more squarely:
- `a_brake_max` 9, `gg_exponent` 5, `curvature_smooth` 21 (ideal 8.295 s on
  the old line);
- the line was re-refined at this profile;
- the corners were raised above 1, one notch below each harness cliff.

The first version (28–43 at 1.2) did 7.99–8.00 s and was **clean on the
full harness grid**, but hit s 30–36 in 3 of 10 judged runs. The judged cliff
there is about 0.1 lower than the harness one. We cut that corner to 1.11
(`a2eafda`): harness 7.97 s, judged 8.03–8.05 s, and then 400+ judged timed
laps with no hit at s 30–36.

### Phase 8: hardening the start of the run (18 Oct, final)

The remaining judged failures were all in the first seconds of a run:

1. **Launch crashes from a clock jump** (`64aa045`). Three runs crashed in the
   first corner. All three began with a ~1 s clock jump that the latch clock
   learnt as RTF 340 (section 4.5). Fix: ignore samples over 10×.
   - We first tried a "launch hold" (stand still until the clock is steady). It
     did not help, which disproved our first theory that the car was "driving
     blind" at launch.
2. **Warm-up contacts at the chicane exit.** They were real near-misses.
   - The harness showed the warm-up lap had the *least* wall clearance of any
     lap: 0.28 m, at the s 40–48 chicane exit.
   - The ~1 s simulator stalls logged around them were an *effect* of the wall
     contact (the simulator stalls on contact), not the cause, which we had
     first suspected.
   - Fix: the slow warm-up (section 4.10, clearance 0.33 m).
3. **A wall pin that disqualified a run** → unstick (section 4.11). Judged runs
   then showed:
   - straight reversal failing → counter-steer;
   - a reverse off the grid → the arming rule.

**Final car, batch T (`CalcuLOL__20261018T004950Z_r1..r10`, commit `ac836d8`):
10 of 10 complete, 0 collisions, 0 wall contacts, best laps
8.038–8.047 s, totals 80.49–80.55 s, RTF 2.28–2.35.** The batch before it,
batch S (the same code without the arming fix), was also 10/10 with 0 contacts.

**Which code batch T ran.** Batch T ran at 08:50–08:59 SGT, from the working
tree, before its changes were committed. They were committed unchanged as
`ac836d8` at 13:11. The evidence:
- the batch log shows the slow warm-up (`0.35 to s 56`, acceleration 0.5) and
  unstick active in all 10 runs;
- batch S's log already shows the counter-steer (`steer +0.42`) and the longer
  repeat reverse;
- the full test suite, run at 08:47 just before batch T, had the same count as
  today (738 passed, 1 deselected), arming tests included.

A single run one minute after the commit (`20261018T051210Z`: 8.008 s, 0
collisions, RTF 1.58) confirmed the committed version.

**Every run of the final code** (30 runs, all `COMPLETE`, 0 collisions in 300
timed laps):

| Runs | Best lap | Collisions |
|---|---|---|
| Batch T, `20261018T004950Z_r1..r10` | 8.038–8.047 s | 0 |
| `20261018T051210Z`, `20261018T092833Z` | 8.008, 8.019 s | 0 |
| `unnamed_team__20261018T074855Z_r1..r5` | 8.009–8.025 s | 0 |
| `unnamed_team__20261018T082903Z_r1..r10` | 8.041–8.049 s | 0 in 9 runs; **r6: 4, all before the timed laps** |
| `20261018T102653Z_r1..r3` (the submitted runs, on the final commit) | 8.036–8.045 s | 0 |

The two `unnamed_team` batches ran the final code, just without `--team`. No
driver logs were kept for them, so we cannot say where r6's contacts happened
or why. **Contacts before the timed laps are reduced, not
eliminated.**

---

## 8. How we test and measure

### 8.1 The offline harness: a faithful copy of the judge

Running the real simulator takes about a minute per run and gives one number.
To find problems, we needed something faster that tells us *why*. But a harness
that is not faithful is worse than none: it sends you tuning in the wrong
direction (Phase 1).

| File | What is exact | What is approximate |
|---|---|---|
| `plant_sim.py` | It **imports** the simulator's own vehicle code (`vehicle_dynamics_st`, `pid_steer`, `pid_accl`, RK4 integrator), with tiny stubs for two packages that are not installed. No physics is re-typed. | — |
| `scan_sim.py` | The judge's own wall extraction (the same 3333 wall segments), a line-for-line port of its LiDAR ray-casting, its car-body rectangle, and the referee's "once per second" collision counting. Collision **detection** is exact. | Collision **response**: we stop the car's motion into the wall (the judged setting has zero bounce) but do not model friction or spin after impact. So it cannot test recovery from a real pin. |
| `lap_sim.py` | It drives the real `DriverCore` against those, with the judge's timing: data 9 physics steps old (configurable), the command landing after `solve time × RTF`, the 25 ms timer and its 3-2-3-2 step pattern, overrunning solves, and laps counted the way the simulator counts them. | Message queues are modelled as a fixed age, not a simulated network. |

`lap_sim` reports:
- lap times and collisions with their arc positions;
- the error of the steering estimate;
- how often the slew limit and emergency cap fired, and coast counts;
- solve times and fallbacks;
- lateral error from the line;
- peak slip.

Overrides (`--driver KEY=VALUE`, `--mpc KEY=VALUE`) let us sweep any setting
without editing code.

### 8.2 The test grid

A setting was "pinned" only after all of:

1. **Fixed solve time × data age grid:** solve 5, 7 or 12 ms × data 5, 9 or 14
   physics steps old. That is 9 cells, deterministic, used to *rank lap time*.
2. **Measured-timing runs:** real, jittery solve times, used to *screen for
   collisions*. Some crashes (s ≈ 66 in Phase 3) appear only with jitter.
3. **10-lap runs.**
4. **A load stress test:** 24 runs while 12 other jobs load the CPU. At budget
   22 this exposed hits that a quiet machine never showed.
5. **The centreline fallback:** the car must still be clean if the line is
   rejected.
6. **Judged batches** of 10 `evaluate.sh --headless` runs, on mains power, on
   a quiet machine.

### 8.3 Lessons about measuring

- **The harness can be clean while the judge is not** (28–43 at 1.2 in Phase
  7). Judged runs are the final word, and we cut margins where they disagreed.
- **The host machine matters.** Runs disqualified when pytest or review agents
  were running at the same time: RTF dropped to 0.6–1.0, solves overran, and
  the car fell back to old plans. Runs on battery power did the same. Before a
  judged batch we check the machine is on mains power and wait for the load to
  settle (`uptime` load below 0.5 over 1 minute and below 1.0 over 5 minutes).
- **Lap time moves slightly with RTF:** lower RTF ran a little faster (8.008 s
  at RTF 1.58 against ~8.04 s at 2.3). The cause is not proven. Our working
  theory is wall-clock delays the driver does not model, which become more
  simulated-time staleness at high RTF.
- **Keep harness grids to 5 parallel jobs or fewer.** More than about 10, and
  real solve times cross the 15 ms budget, which corrupts the results.

### 8.4 Unit tests

**30 test files, 739 tests**, in `test/`. The last measured coverage was about
95% (Oct 9). Run them with:

```sh
cd race_ws/src/team_driver
OMP_NUM_THREADS=1 python3 -m pytest test -q -p no:cacheprovider \
  --deselect test/test_dt_schedule.py::UniformScheduleReproducesTodayTest::test_closed_loop_commands_are_the_pre_step3_controllers_to_the_bit
```

The deselected test is a bit-identity check known to differ by ~1e-6 under
threading; it was already failing before these changes. **Never run tests
during `evaluate.sh`:** the load causes crashes.

What the tests protect:

| Group | Files | What they check |
|---|---|---|
| Fidelity to the judge | `test_plant_sim`, `test_scan_sim`, `test_st_model`, `test_steering_actuator`, `test_speed_loop`, `test_model_error` | Our models and harness match the simulator's code; the servo fit; the speed-loop inversion |
| Harness | `test_lap_sim`, `test_harness_cadence`, `test_harness_overrides`, `test_dead_time` | Timing, staleness, overrides, lap counting |
| MPC | `test_command_memory`, `test_move_blocking`, `test_dt_schedule`, `test_fused_objective`, `test_reference_wrap`, `test_reference_pace`, `test_reference_speed_profile`, `test_latch_predictor`, `test_mpc_braking` | Warm start, blocks, schedule, gradient = SciPy's, wrap-around, pace, braking bound |
| Line and profile | `test_raceline`, `test_speed_profile`, `test_speed_profile_physics`, `test_profile_knobs` | Cone sealing, clearance, g-g, sectors, knobs |
| Safety and wiring | `test_output_conditioning`, `test_step4_safety`, `test_cap_reference`, `test_step3_wiring`, `test_driver_params`, `test_warmup_mode`, `test_unstick` | Slew, coast, caps, warm-up switching, unstick, that the YAML equals the code, that bad values are refused, and that the upstream vehicle constants have not changed |

---

## 9. Things we tried that did not work

Each row is a real attempt, with what happened and why.

| Tried | What happened | Why |
|---|---|---|
| Tuning MPC weights against the old harness | Nothing improved | The harness had no servo and no delay, so it could not see the chatter (Phase 1). |
| Lower `r_delta_steer` (0.1–0.8) | More chatter, slower (+0.7 s/lap at 0.1) | Steering changes too cheap. |
| A longer MPC horizon | Solve time exploded (12 steps: 139 ms) | Cost grows faster than the step count; we stretched the step lengths instead. |
| `raceline_margin` 0.10 / 0.20 | A disqualification at 0.10; a collision at s ≈ 65.5 at 0.20 | Too little clearance at speed. |
| Braking harder than 6 m/s² with the old controller | Spins | Braking yaw instability (section 2.5); it worked only after the MPC owned the braking. |
| g-g exponents 3–4 with the old controller | Spins | Same reason. |
| Uniform budget 23–27 with the old braking | 63 of 63 harness runs spun | Same reason (Phase 6). |
| `profile_brake_cap` 0 (MPC-only braking) | Clean up to 26, lost the chicane exit at 27 | The car needs a little profile braking there. |
| A yaw-rate tracking cost in the MPC | Crashed the out lap at every weight | Dropped. |
| `beta_limit` 0.5 at high budgets | A spin at s ≈ 16 | The car really slides past 0.5 rad; hiding that from the MPC is wrong. |
| `a_brake_max` 9.5 with exponent 6 / smoothing 25 | A collision at s 54 | Too aggressive. |
| Budget 28+ | Hit the chicane exit (s 45.4) | Past the cliff. |
| Corner 28–43 at 1.2 | Harness-clean, judged-unsafe (3/10) | The harness and the judge differ near the cliff. |
| "Launch hold" against launch crashes | No effect | The real cause was the clock jump. |
| "Host stalls cause the warm-up hits" | Wrong theory | The stall followed the contact; the cause was a near-miss on the warm-up line. |
| Straight reversal in unstick | Nose back on the wall, disqualified | Replaced with counter-steer. |
| Unstick at 0.3 m/s | Never fired | A scraping car creeps at 0.37–0.5 m/s. |
| Budget 22 at sub-10 (instead of 21) | Clean on the grid, but 2/18 hits under load | Kept 21 for safety. |
| Using the mid-step wheel angle in the ST model | Better open-loop prediction, worse closed-loop (2 collisions vs 0) | The start-of-step angle's slight lag offsets unmodelled delay. Kept as is. |

---

## 10. Known limitations, risks and failure modes

Where the car can still break, and why.

### 10.1 Risks on judging day

- **Host load / low RTF is the main risk.** On a loaded or throttled machine
  (RTF near 1 or below), solves overrun and data gets older than our tests
  assume. Runs like that were disqualified on our machine. The judging machine
  (i9-14900HX) is faster than ours, so we expect RTF at or above ours. Best of
  3 and the slow warm-up reduce this risk; they do not remove it.
- **Contacts before the timed laps are not fully eliminated.** 1 of 30 runs
  of the final car had 4 collisions before the timed laps (`082903Z_r6`, RTF
  2.02). They cost no time but count towards disqualification. With no log,
  the cause is unknown.
- **RTF above about 2.4 is untested.** Above 10× real time our clock filter
  would ignore every sample and assume RTF 1, which would make latch times
  early. We have no data there.
- **Unstick is not proven on a hard pin in the real simulator.** It has
  escaped real contacts in judged runs (one reverse), and it is unit-tested,
  but our harness's contact model cannot reproduce a real pin, and we never
  forced one.
- **The GitHub Judge workflow** runs on slow shared hardware. It may show
  contacts or low RTF; it is for reference only and not scored.

### 10.2 Design limitations

- **The sectors are specific to this track and this line.** A new track needs a
  re-tune, or the uniform budget at its lowest scale (which the driver applies
  automatically on any untuned line).
- **The MPC knows nothing about walls.** Safety comes from the line's
  clearance and the emergency cap. A car pushed far off the line relies on the
  cap alone.
- **The emergency cap cannot see a full stopping distance at top speed.**
  Stopping from 12 m/s at 7 m/s² takes 10.3 m, but LiDAR readings are clipped
  at 8 m. It is a last-resort net that relies on the line not pointing the car
  at a wall, not a guarantee.
- **The latch clock does not model transport delay** (the time a message spends
  between programs). That is a likely cause of the small lap-time shift with
  RTF.
- **We use ground-truth odometry**, not our own localisation.
- **The tyres in this simulator never lose grip.** Our slip budgets (27 m/s²)
  would be impossible on a real car. The approach would transfer, but not the
  numbers.

### 10.3 The fallback chain (what happens when something fails)

| What fails | What happens |
|---|---|
| The shipped line fails its map check | Build a min-curvature line from the map (~3 s); raised sectors clamped to 1 |
| That build fails | The cached CSV (if safe), else the centreline at a uniform, cautious budget |
| Map files not found | Follow-the-gap at 3 m/s |
| The warm-up profile fails | Race from the grid |
| An MPC solve is too slow or fails | Follow the previous plan one step on |
| The MPC throws an error, or a NaN appears | Coast: unwind the steering, shed speed at 7 m/s² |
| No scan yet | Coast (standing still) |
| Car pinned on a wall | Counter-steer reverse, longer on repeats |
| Simulator reset | All memory cleared; warm-up restarts |
| A bad config value | The node refuses to start, with a message naming the value |

### 10.4 What we would do with more time

- A faster minimum-time solver, so the whole line can be computed at startup
  (the full "runtime line" bonus).
- Our own localisation (for example, a particle filter matching LiDAR scans to
  the map).
- Modelling transport delay in the latch clock.
- Making the MPC robust to simulator stalls (time-stamped command plans).
- A better tyre/load-transfer model in the MPC, to replace the per-corner
  sectors.

---

## 11. The rules: how we comply

### 11.1 Rule 32: precomputation and "hand-tuned waypoints"

- **The racing line is computed by code from the map alone.** No waypoint was
  placed or edited by hand. Rebuild it with:
  ```sh
  OMP_NUM_THREADS=1 python3 race_ws/src/team_driver/team_driver/raceline.py \
      --margin 0.25 --refine --write race_ws/src/team_driver/config/icra26_raceline.csv
  ```
  It takes about 4 minutes and reproduces the shipped file.
- The refinement's speed settings (`REFINE_PROFILE`) are a fixed constant in
  `raceline.py`, so the result depends only on the map and the code.
- The driver does not blindly trust the file: it checks every point against
  the map at startup.

### 11.2 The sector scales: parameters, not waypoints

They are the part that looks most like hand-tuning, so here is exactly what
they are:

- They are **parameters** (multipliers on a speed budget along stretches of
  the line), not waypoints. They do not move the line.
- Each was set by **measurement**: we swept the budget on that stretch in the
  offline harness and recorded where collisions happened. Each raised corner
  sits one step below the value that collided. The 28–43 corner was lowered
  further because judged runs disagreed with the harness.
- They are **tied to this line**. The driver clamps them (no raises above 1)
  on any other line, and uses a uniform cautious budget on the centreline.
- The numbers and their evidence are documented in `DriverConfig` and
  `driver_params.yaml`.

### 11.3 Other rules

- **Rule 11 (cones):** the line builder seals cone gaps before planning
  (section 4.3).
- **Rule 25 (judging environment):** we changed nothing in the protected
  paths. Our work is in `race_ws/src/team_driver/`, plus `SUBMISSION.md`,
  `results/submitted/` and our own helper notes in `dev-scripts/`.
  `./scripts/verify_judging_env.sh` checks this. The offline harness *reads*
  the simulator code in `external/` but never modifies it.
- **Rule 28:** we publish only `/drive` and `/driver/markers`.
- **Rule 31:** ground-truth odometry, as allowed.
- **Rule 33 (dependencies):** `numpy==2.5.3`, `scipy==1.18.1`, pinned in
  `requirements.txt`. No apt packages.
- **Rule 34:** the entry point `ros2 run team_driver driver` is kept.
- **Rule 35 (real-time):** solves are bounded by the 15 ms budget.
- **Rules 36–38 (AI):** we used Claude Code (Anthropic) heavily for
  implementation, tests, the harness and run analysis. The team set the
  direction, chose between measured hypotheses, ran the judged evaluations and
  decided what shipped. This document exists so every member can explain every
  part.

---

## 12. Every file, and why it exists

The organisers' template gave `driver.py` (a stub), `setup.py`, `setup.cfg`,
`package.xml`, `launch/driver.launch.py`, an empty `config/driver_params.yaml`,
`requirements.txt` and `apt-packages.txt`. Everything else is ours.

### Runs during the race

| File | Lines | What it does |
|---|---|---|
| `team_driver/driver.py` | 246 | The ROS node: parameters, subscriptions, the 25 ms timer, publishing, the latch clock, the RViz arrow, health logs. No algorithm. |
| `team_driver/driver_core.py` | 1536 | The algorithm, ROS-free: `DriverConfig`, line loading and checks, the steering estimator, the latch clock, warm-up, unstick, `plan` (MPC + gap-follower), the emergency/profile caps, conditioning and coasting. (Large because it holds the whole control path, one cohesive unit.) |
| `team_driver/driver_params.py` | 208 | The single parameter table: declare, read, validate, echo. |
| `team_driver/mpc_controller.py` | 1092 | The MPC: reference window (arc tracking, reachable pace, wrap-around), cost, fused objective and gradient, SLSQP solve, warm start, budget and fallback. |
| `team_driver/vehicle_model.py` | 591 | The vehicle constants, ST and KS models (exact lateral step), the servo model, the simulator's speed loop and its inversion. |
| `team_driver/speed_profile.py` | 307 | Corner, lock, braking (g-g) and acceleration passes; sectors; the ideal lap time. |
| `team_driver/latch_predictor.py` | 134 | Replays published commands through the plant to the next latch. |
| `team_driver/longitudinal.py` | 188 | Geometry for the speed caps: the closed path by arc length, corridor clearance from LiDAR points, the bumper overhang. |
| `team_driver/raceline.py` | 642 | Map loading, cone sealing, clearance, the min-curvature QP, the min-time refinement, CSV reading/writing, the command-line tool. |
| `config/icra26_raceline.csv` | 744 points | The shipped min-time line (0.25 m margin). |
| `config/driver_params.yaml` | — | Every value, equal to the code defaults, with comments recording the evidence. |
| `requirements.txt` | — | `numpy==2.5.3`, `scipy==1.18.1`. |
| `launch/driver.launch.py` | — | Starts the node with the YAML (for experiments). |

### Offline tools (never run in a race)

| File | What it does |
|---|---|
| `team_driver/plant_sim.py` | The judge's car physics, imported (not copied) from `external/f1tenth_gym`. |
| `team_driver/scan_sim.py` | The judge's walls, LiDAR, body and collision counting; an approximate contact response. |
| `team_driver/lap_sim.py` | The closed-loop harness with the judge's timing; overrides; JSON output. |
| `team_driver/lap_floor.py` | The ideal lap time of a line at several budgets: how much time is physically available. |
| `team_driver/model_error.py` | How far each MPC model's prediction drifts from the plant. |
| `team_driver/mpc_controller_simple.py` | An early prototype. **Nothing imports it**; it is kept for history only and could be deleted. |
| `.coverage` | A leftover coverage data file; safe to delete. |

### Tests

`test/`: 30 files, 739 tests (section 8.4).

---

## 13. Reproducing everything

```sh
# Inside the container (/hackathon is the repo)
cd /hackathon/race_ws && colcon build --symlink-install && source install/local_setup.bash

# Judged-style runs (machine on mains power and idle!)
cd /hackathon && ./scripts/evaluate.sh --team CalcuLOL --runs 3 --headless

# Offline harness: 10 laps under judged timing
cd /hackathon/race_ws/src/team_driver/team_driver
OMP_NUM_THREADS=1 python3 lap_sim.py --laps 10 --json /tmp/r.json
# one setting changed, e.g.
OMP_NUM_THREADS=1 python3 lap_sim.py --laps 3 --fixed-solve-ms 7 --driver a_lat_max=26

# The physical floor of the line at a few budgets
python3 lap_floor.py --path ../config/icra26_raceline.csv --a-lat 21 24 27

# Regenerate the racing line (single-threaded, ~4 min)
OMP_NUM_THREADS=1 python3 raceline.py --margin 0.25 --refine --write ../config/icra26_raceline.csv

# Check the judging environment is untouched
cd /hackathon && ./scripts/verify_judging_env.sh
```

The startup log shows which configuration and line ran: `params_source =
CODE_DEFAULTS`, then `raceline: offline min-time line …, 744 points`.

---

## 14. Things in the code that may look wrong

Worth knowing if you read the code. None changes behaviour.

- **Library defaults are not the race car.** `VehicleModel` defaults to the
  kinematic model `'KS'`, and `MPCController` defaults to horizon 10 and
  `r_delta_steer` 0.8. These are only the constructors' defaults. The node and
  the harness always pass the race values from `driver_params.py` (`ST`,
  horizon 8, `r_delta_steer` 3.0); `test_driver_params.py` and
  `test_step3_wiring.py` hold them equal.
- **`mpc_controller.py` prints** its configuration with `print` at startup
  instead of the logger. This is harmless; it shows in the node's console.
- **Two centreline-fallback lap times appear in the comments**: 10.97 s and
  9.37 s. They are from different tunes, not a contradiction. 10.97 s is the
  sub-9 Step 2 tune (older braking shape). 9.37 s is today's profile, whose
  ideal centreline lap is 9.49 s; harness laps usually come in just under the
  ideal. Both are at the same uniform 22.95 m/s² budget.
- **Comments that were stale and fixed on 18 Oct** (comment-only; the code was
  checked to be identical):
  - the `unstick` comments said "reverse straight", but the code counter-steers;
  - `mpc_controller.py`'s header said "kinematic bicycle model";
  - `lap_sim.py`'s header described the old depth-10 queues as current;
  - `DriverConfig` said warm-up hits came in "3 of 13" sub-9 runs. The result
    files show 2 of 13 runs, with 3 hits in total.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Arc length (s)** | Distance along the racing line from its first point, in metres. We name places on the track by it. |
| **base_link** | The car's reference point: the rear axle. |
| **Block / move blocking** | Holding one MPC decision over several prediction steps, to cut the number of variables. |
| **Coast** | Our safe "no plan" behaviour: unwind the steering, brake gently. |
| **Cost function** | The score the MPC minimises: errors plus effort. |
| **Curvature (κ)** | How sharply a path bends: 1 / radius. A straight has κ = 0. |
| **Dead time** | The delay between what the data describes and when our command acts (~0.1 s). |
| **DNF / DQ** | Did not finish / disqualified (more than 10 collisions). |
| **Emergency cap** | The LiDAR-based speed limit: a speed from which the car could still stop before the nearest wall in its path. |
| **Finite differences** | Estimating a slope by nudging an input slightly and seeing how the output changes. |
| **g-g limit** | Sharing one acceleration budget between braking and cornering. |
| **Horizon** | How far ahead the MPC predicts (8 steps, 0.5 s). |
| **Ideal lap** | The time to drive the line exactly at its speed profile. |
| **Kinematic (KS) model** | A car model that assumes no sideways sliding. |
| **Latch** | The moment the simulator picks up a new command and starts acting on it. |
| **L-BFGS-B** | A standard optimiser for smooth problems with box limits; used for the line refinement. |
| **LiDAR** | A laser range finder: distances to walls in many directions. |
| **Load transfer** | Weight moving to the front tyres under braking (to the rear under acceleration). |
| **MPC** | Model Predictive Control: plan ahead with a model, apply the first step, repeat. |
| **Normal** | The direction at right angles to the track at a point. |
| **Odometry (odom)** | The car's position, heading and velocity. Here it is ground truth from the simulator. |
| **Plant** | Control-engineering word for "the thing being controlled": here, the simulated car. |
| **Proportional controller** | A controller whose output is a gain × the error (the simulator's speed loop and servo). |
| **QP / least squares** | Optimisation where the cost is a sum of squares; our min-curvature line is one. |
| **Real-time factor (RTF)** | Simulated seconds per wall-clock second. 2.3 means the simulator runs 2.3× faster than real time. |
| **Reference (window)** | The target states along the racing line that the MPC tracks. |
| **ROS 2 / node / topic** | The robot messaging framework; a program; a named message channel. |
| **RViz** | The ROS 3D visualiser. |
| **Sector** | A stretch of the line where the slip budget is scaled. |
| **Servo** | The steering motor: it moves the wheel towards the asked angle at a limited rate. |
| **Sideslip (β)** | The angle between where the car points and where it moves. |
| **Single-track (ST) model** | The "bicycle" car model with tyre slip, used by the simulator and our MPC. |
| **Slew limit** | A cap on how much a command may change per step. |
| **Slip budget (`a_lat_max`)** | Our cornering limit in m/s². With linear tyres it is a controllability limit, not grip. |
| **SLSQP** | Sequential Least-Squares Programming: the optimiser our MPC uses. |
| **Speed profile** | The target speed at every point of the line. |
| **Understeer** | Needing more steering at speed than geometry alone suggests. |
| **Warm start** | Starting an optimisation from the previous answer instead of from scratch. |
| **Yaw / yaw rate** | The car's heading / how fast it rotates. |
