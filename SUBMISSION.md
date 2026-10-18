# Team CalcuLOL

## Members

| Name | Matric number | Email |
| --- | --- | --- |
| Anish Gangavaram | U2523701F | GANG0016@E.NTU.EDU.SG |
| Davis Lee | U2523351H | DLEE067@E.NTU.EDU.SG |
| Myra Dwivedi | U2523271G | myra.ofc@gmail.com |

## Approach

Our driver is a **nonlinear Model Predictive Controller (MPC)** that follows a racing line generated from the circuit map. Every control cycle (25ms), it estimates where the car will be when its next command takes effect, plans its motion over the next 0.5 s, then applies safety limits. The controller uses a dynamic single-track vehicle model, with the steering servo represented as a state. It optimises steering and acceleration with SLSQP over 8 prediction steps, but shares only 3 control decisions between those steps (“move blocking”) to keep the solve within a 15 ms wall-clock budget. Because odometry and scans are stale and commands take effect later, the driver replays its own commands through the simulator’s servo, speed controller and vehicle dynamics, then plans from the predicted latch-time state.
The main implementation is in `mpc_controller.py`, `vehicle_model.py`, `latch_predictor.py` and `driver_core.py`.

We started with Follow-the-Gap (best lap 30.36 s) and Pure Pursuit (17.83 s). We also tried Pure-Pursuit with a pre-computed racing line but these algorithms had diminishing returns in their further optimisation. Both react to where the car is now, but what limits this car is what happens next: a command acts about 0.1 s late, the wheels take time to turn, and the car slides in corners. MPC can plan for all three, but only if its model matches the simulator. We therefore matched the simulator’s vehicle parameters, steering servo and speed controller, and added tests that compare our implementations against the simulator code. We tuned the controller in an offline harness that reproduces the judged timing. The harness uses the same `DriverCore` as the ROS node and imports the judge simulator’s vehicle plant and wall extraction where possible. This lets us compare predicted and actual behaviour, identify root causes, and screen changes before judged runs. The measured progression was from a slow centreline baseline to about 12.56 s, 11.08 s, 10.87 s, 9.87 s, 8.82 s, and finally about 8.04 s per lap. The changes and corresponding commits are recorded in the evidence table below.

The important improvements came from fixing failure mechanisms rather than simply increasing speed. Instrumentation found that early high-speed crashes were braking-induced yaw: the profile could demand −5 to −7.8 m/s² while the MPC model was limited to −2.85 m/s². We moved braking ownership into the MPC and introduced a combined lateral/longitudinal grip limit. We also fixed a startup clock jump that made the steering estimate badly stale, added a slower warm-up to reduce exposure to early-run stalls, and added a counter-steer unstick behaviour for sustained wall contact. These decisions were checked in the offline harness and then against judged runs. We keep known limitations explicit: performance remains sensitive to host load and timing stalls, and the corner-specific speed margins are tuned for this circuit.

### Evidence of engineering decisions

| Change or decision | Evidence and outcome |
| --- | --- |
| Use one `DriverCore` for both ROS and the offline harness | `driver_core.py` contains no ROS wiring; `driver.py` is the ROS entry point. The harness drives the same core, avoiding the earlier mismatch where the harness supplied a different steering command. |
| Fix MPC reference pacing | Pacing the reference window at car speed silently limited acceleration; changing it produced a 12.56 s lap (`525ce12`). |
| Add sector tuning and retune grip budget | Improved the judged lap to 11.08 s (`7489614`). |
| Ship an offline-refined minimum-time line | Improved the judged lap to 10.87 s (`c15552a`). |
| Add combined g-g braking and a path-following safety cap | Addressed braking-induced yaw and false emergency-cap activations; reached 9.87 s with zero collisions (`186026c`, `6c3d650`). |
| Make the MPC own braking | Reached 8.82 s, with zero collisions in 130 timed laps; 3 of those 13 runs touched a wall during the untimed warm-up, which led to the slower warm-up below (`2fd33e5`). |
| Refine profile shape and raise measured corner budgets | Reached approximately 8.03–8.05 s (`a2eafda`). |
| Harden startup clock, warm-up and unstick behaviour | Batch T, run after the clock fix (`64aa045`), completed 10/10 runs with zero collisions. The counter-steer unstick and slower warm-up (`ac836d8`, the submitted HEAD) were then checked in a single run (`20261018T051210Z`). |

Several alternatives were rejected based on observed failure, not preference: adding a yaw-rate tracking cost crashed the out-lap at every tested weight; a launch hold did not fix startup crashes because the root cause was the clock; stronger braking and aggressive grip budgets caused spins with the old braking controller; a longer MPC horizon took 139 ms at 12 steps; and a 0.20 m line margin collided near s ≈ 65.5 m. The reasoning behind each pinned value is in the comments of `config/driver_params.yaml` and `DriverConfig` in `driver_core.py`, and in the tests.

## How it uses the inputs

- **LiDAR:** used for safety and fallback behaviour, not to generate the racing line.
  1. The scan is placed relative to the car and used to estimate clearance in a car-wide corridor that follows the planned path/reference.
  2. The emergency speed cap accounts for dead-time travel and compares available clearance with braking distance.
  3. If no usable reference is available, a reactive gap-follower provides a fallback.

  The emergency cap follows the planned path rather than looking only straight ahead, which avoids braking for the outside wall of every corner. See `driver_core.py`, `longitudinal.py` and `scan_sim.py`.

- **Odometry:** the simulator's ground-truth `/ego_racecar/odom` topic, which the rules explicitly permit. We did **not** build our own localisation system. Position, heading, velocity, sideslip and yaw rate initialise the dynamic single-track model. Yaw rate also constrains the predicted steering angle because the wheel angle is not directly measured. See `driver.py`, `driver_core.py` and `latch_predictor.py`.

- **Map:** the racing line is generated and validated from the circuit files in `maps/` (the occupancy grid and centreline CSV, read from disk rather than from the `/map` topic). Cone rows are sealed before optimisation so the line cannot thread gaps between cones, as required by rule 11. At startup, the driver accepts the shipped line only if its header identifies the `min-time` method and every point clears the sealed map by at least 0.405 m (0.155 m half-width plus 0.25 m margin). If validation fails, it builds a minimum-curvature line from the map; the centreline is the final fallback. The LiDAR scan does not alter the racing line.

## Results on our own machine


| Result set | Best lap | 10-lap total | Collisions | Runs attempted |
| --- | ---: | ---: | ---: | ---: |
| Batch T (`20261018T004950Z`, 10 runs) | 8.038–8.047 s | 80.49–80.55 s | 0 in every run | 10 (10 COMPLETE) |
| HEAD run (`20261018T051210Z`) | 8.008 s | 80.25 s | TODO: confirm from its JSON | 1 |

TODO before submitting: commit  result JSONs to `results/submitted/`, unedited. That folder currently holds only an older batch (`20261017T085529Z`, 9.87 s, the sub-10 car) — either keep it and say so here, or remove it.

### The machine that produced them

| CPU | GPU | RAM | OS | Typical real-time factor |
| --- | --- | --- | --- | --- |
| Intel Core Ultra 7 155H (22 threads) | NVIDIA RTX 4050 Laptop, 6 GB (unused by driver) | 31 GiB visible to WSL | Windows 11 + WSL2 Ubuntu 24.04, kernel 6.18.33.2, Docker Desktop | 2.28–2.35 for Batch T |

The MPC solve budget is measured in wall-clock time, so host load can reduce the time available to plan and can change when commands latch. The clean runs above were made with the host idle and on AC power. We have also observed disqualifications on loaded or throttled runs near RTF 1.0; see **Known issues**.

## Anything precomputed

The shipped racing line is:

`race_ws/src/team_driver/config/icra26_raceline.csv`

It is generated by `race_ws/src/team_driver/team_driver/raceline.py` from the map and centreline data. No waypoint was placed or edited by hand, and the map was not modified. The line-generation pipeline is:

### 1. Minimum-curvature line

The script seals cone rows, calculates lateral clearance bounds along centreline normals, and solves a box-bounded least-squares problem that minimises squared second differences. It re-linearises the problem over several rounds and re-weights curvature so tight apexes remain drivable. This follows the minimum-curvature method described by Heilmeier et al. The minimum-curvature line can also be built at runtime if the shipped line fails validation.

### 2. Offline lap-time refinement

The faster shipped line is refined offline using 50 lateral-offset control points and a periodic cubic spline. L-BFGS-B minimises the ideal lap time of the speed profile while penalising clearance violations, over three re-linearised rounds. In the recorded `REFINE_PROFILE` run, predicted ideal lap time improved from 8.169 s to 8.009 s. The refinement takes about four minutes, which is too slow to run every time the driver starts, so the refined line is shipped as data. The runtime minimum-curvature build remains as the fallback.

Regenerate the line with:

```sh
OMP_NUM_THREADS=1 python3 race_ws/src/team_driver/team_driver/raceline.py \
  --margin 0.25 --refine \
  --write race_ws/src/team_driver/config/icra26_raceline.csv
```

`OMP_NUM_THREADS=1` is important for reproducibility. We measured that threaded BLAS can move the initial QP line by about 1e-6 m; the refinement can amplify this into a difference of about 9 cm. The refinement profile is a constant in `raceline.py`, so the result is reproducible from the map and submitted code alone.

The speed profile is computed by forward/backward passes under lateral-slip, steering-lock, acceleration and braking limits. Corner-specific sector scales were selected through measured sweeps, then confirmed in judged runs. They are not hand-entered waypoints: they are multipliers on the slip budget for arcs of this particular line, and the driver clamps raised sector values when using a line they were not tuned for. The relevant parameters and their rationale are documented in `config/driver_params.yaml` and `driver_params.py`.

Nothing is trained; there are no learned weights.

## Dependencies we added

| Package | pip or apt | What we use it for |
| --- | --- | --- |
| `numpy==2.5.3` | pip | Arrays used by the driver, speed profile and offline harness. |
| `scipy==1.18.1` | pip | Least-squares line optimisation, SLSQP MPC optimisation, L-BFGS-B refinement, cubic splines and image processing for cone sealing. |

Both are declared in `race_ws/src/team_driver/requirements.txt` and pinned for reproducibility. No apt packages are declared. Other packages used by the driver, including `rclpy`, are already available in the judging image.

## Third-party code and references

1. **Heilmeier et al. (2019).** “Minimum curvature trajectory planning and control for an autonomous race car.” *Vehicle System Dynamics*. We implement a minimum-curvature method in `raceline.py`.
2. **F1TENTH Gym.** The pinned simulator in `external/`. The offline harness imports the simulator's vehicle plant, steering servo, integration and single-track dynamics in `plant_sim.py`; `scan_sim.py` uses the judge's wall extraction and a line-for-line port of the LiDAR kernel.
3. **Vehicle-dynamics practice.** The combined lateral/longitudinal (“g-g”) braking limit uses the friction-ellipse idea to reduce braking demand as lateral grip is consumed.
4. **Organiser documentation.** The algorithms and judging documentation guided the initial Follow-the-Gap and Pure Pursuit baselines and the evaluation process. The cone-sealing step reimplements the idea behind `scripts/track_tool.py --link-cones`.

## AI assistance

Claude Code (Anthropic) was used to help implement and review the core-logic of our driver, write tests, build the offline harness and probe scripts. It was also used verify changes that we made after our analysis of test-run results. The team chose the direction of the work, compared measured hypotheses, ran judged evaluations, and decided which changes to ship. ChatGPT and Claude Code were used to help organise, edit and fact-check this submission document against the code. 

## Known issues

1. **Host load and timing stalls.** This is the main remaining risk. On a loaded or throttled host, the simulator can advance for a long interval on a stale command and the car can contact a wall. Runs around RTF 1.0 have been disqualified. The submitted clean batch was run on AC power with an idle host. Best-of-three judging and the slower warm-up reduce, but do not remove, this risk.
2. **RTF affects measured lap time.** A run at RTF 1.58 recorded 8.008 s, compared with about 8.04 s at RTF 2.3. Sustained RTF above about 2.4 has not been tested. The judging hardware differs from ours, so local times are not a guarantee of judged times.
3. **Unstick is not fully validated on hard pins.** The behaviour is unit-tested and has been exercised in judged contact cases, but the offline harness approximates wall contact. Counter-steering is intended to reduce repeated contact; it is not a guarantee of recovery.
4. **Sector scales are track-specific.** They were tuned for `icra26` and the shipped line. A different track needs retuning; on a line not recognised as the tuned line, raised sectors are clamped.
5. **No dynamic obstacle avoidance.** LiDAR limits speed and supports the gap-follower fallback, but does not re-plan the racing line around an obstacle.
6. **No own localisation.** The driver relies on permitted ground-truth odometry rather than scan matching or a particle filter.

**What we would do next.** The controller is tuned for the `icra26` circuit and the simulator's tyre model. The most useful next steps would be to make minimum-time refinement fast enough for startup, model transport lag in the latch predictor, make the MPC robust to long simulation stalls with a time-stamped command horizon, replace per-corner sector tuning with a better tyre/load-transfer model, and investigate scan-matching localisation. None of these is claimed as implemented.
