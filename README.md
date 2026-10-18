# CalcuLOL: an MPC driver for autonomous racing

This is our entry for Track 1 of the NTU DeepSpeed Recruitment Hackathon 26/27. The task was to write the software that drives a 1/10-scale car around a circuit as fast as possible without hitting anything. Our final driver does a lap in about **8.04 s with no collisions**.

Team CalcuLOL: Anish Gangavaram, Davis Lee and Myra Dwivedi. I wrote most of the driver (the MPC, the offline test harness, the racing line and speed profile, and the debugging and tuning that got the lap time down). Davis and Myra handled the submission write-up and checks.

## How we got there

We started with the simple approaches. Follow-the-Gap managed 30.4 s and Pure Pursuit got to 17.8 s, but both only react to where the car is right now. On this car that's the wrong question: commands take effect about 0.1 s late, the steering takes time to turn, and the car slides in corners. So we switched to model predictive control.

Our first MPC was actually slower than Pure Pursuit (around 50 s, with crashes). The model didn't match the simulator. Once we matched the simulator's vehicle parameters, steering servo and speed controller, and built an offline harness that reproduced the judged timing, the lap time came down step by step: 18.0, 12.6, 11.1, 10.9, 9.9, 8.8 and finally 8.04 s.

Most of the gains came from finding out *why* the car crashed rather than just turning the speed up. For example, almost every high-speed crash turned out to be the rear end stepping out under braking. The fix was to let the MPC take over braking instead of following a fixed speed profile.

## What the driver does

- Every 25 ms it predicts where the car will be when the next command actually takes effect, and plans from there.
- A nonlinear MPC plans 0.5 s ahead, solved with SLSQP inside a 15 ms budget.
- It follows a minimum-time racing line that is optimised offline and shipped as data.
- The speed profile limits braking when the car is already using a lot of grip in a corner.
- LiDAR is used for safety checks, and there's a recovery move if the car ends up against a wall.

The code is in [`race_ws/src/team_driver/team_driver/`](race_ws/src/team_driver/team_driver/). The main files are `mpc_controller.py`, `driver_core.py`, `latch_predictor.py` and `vehicle_model.py`.

For the full story, [HOW_IT_WORKS.md](HOW_IT_WORKS.md) walks through the whole driver, and [SUBMISSION.md](SUBMISSION.md) has the results and the evidence for each change.

This is a copy of our team repo, [davislee23/DeepSpeedHackathon_CalcuLOL](https://github.com/davislee23/DeepSpeedHackathon_CalcuLOL). It's built on the simulator and template that the NTU DeepSpeed. Their original challenge brief is below.

---

<details>
<summary><b>Original hackathon brief (from the organisers)</b></summary>


# RoboRacer Autonomous Racing — Track 1

**NTU DeepSpeed Recruitment Hackathon 26/27**

Write the software that drives a 1/10-scale racing car around a circuit it has
never seen, as fast as it can, without hitting anything. Everything you need —
the simulator, the judging environment, reference algorithms and every official
track — is in this repository. You will not need to clone anything else.

> **Submission deadline: 18 October 2026, 23:59 (SGT)**

---

## Quick start

```sh
git clone --recurse-submodules https://github.com/NTUDeepSpeed/Recruitment_Hackathon_2627.git
cd Recruitment_Hackathon_2627
git checkout track1

./install/linux/setup.sh          # or install/macos, or install/windows from WSL
./install/linux/run.sh            # builds if needed, then drops you into the container
```

Then, inside the container:

```sh
cd /hackathon/race_ws && colcon build --symlink-install
source install/local_setup.bash

# Terminal 1 — the simulator
ros2 launch roboracer_referee simulator.launch.py

# Terminal 2 — your car
ros2 run team_driver driver
```

The car will set off and hit the first wall: the template is wiring, not a
driver, and writing one is the hackathon. Once yours does something, score
it:

```sh
./scripts/evaluate.sh --team your_team_name
```

Pushing also races your entry on GitHub Actions and writes the result to the
workflow summary. That workflow is for reference only — it runs on GitHub's
hardware and nothing it prints is scored. **Your result is the organisers' run
on the judging machine.**

Full walkthrough: **[docs/01-setup.md](docs/01-setup.md)**.

> Cloned without `--recurse-submodules`, or downloaded a ZIP? Run
> `git submodule update --init --recursive`. The setup scripts will also do it
> for you. A ZIP download will not work — submodules need a real clone.

---

## The challenge

You get a LiDAR scan and the car's exact position. You publish a steering angle
and a speed. That is the whole interface.

| | |
| --- | --- |
| **Car** | Ackermann steering, 0.33 m wheelbase, 0.31 × 0.58 m, steering limited to ±0.4189 rad |
| **Sensor** | 819-beam LiDAR, 270° field of view, 25 m range |
| **Localisation** | Ground-truth pose on `/ego_racecar/odom` — **allowed and recommended** |
| **Track** | `icra26`, in this repository. 17 x 18 m, about 78 m a lap, cone slaloms and a hairpin complex. |
| **Scored on** | Your single fastest lap, and your time for 10 consecutive laps |
| **Penalties** | +10 s on the lap for each collision; more than 10 collisions is a disqualification |
| **Judged on** | One machine: i9-14900HX, 32 GB, RTX 5060 Laptop. Times are in simulated seconds, so your own hardware does not affect your score. |

**No driver ships with this repository.** `team_driver` is wiring — it talks to
the simulator and drives in a straight line into the first wall. Writing
something that laps is the hackathon. Reactive algorithms like follow-the-gap
will get you round; planning against the map and your own position is where the
lap time is; and **replacing the approach outright is encouraged**:
reinforcement learning, MPC, imitation learning, a learned end-to-end policy,
anything you can defend. The environment is ROS 2 Jazzy on Python 3.12 with a
GPU available, so a learned policy is a realistic option.
[docs/04-algorithms.md](docs/04-algorithms.md) is the menu, with what each
approach needs and where each breaks.

---

## Where things are

```
Recruitment_Hackathon_2627/
├── install/                    Setup scripts, one folder per platform
│   ├── linux/  macos/  windows/    setup.sh, run.sh, shell.sh, stop.sh
│   └── common.sh
├── race_ws/src/                The ROS 2 workspace, mounted into the container
│   ├── team_driver/            ★ YOUR CODE GOES HERE — the template does not drive
│   └── roboracer_referee/        The judging environment — do not modify
├── scripts/                    evaluate.sh, leaderboard.py, track_tool.py, …
├── maps/                       The circuit: icra26.pgm, its yaml, tracks.yaml
├── docker/                     Image definition (ROS 2 Jazzy) and compose files
├── external/                   Upstream simulator sources, as git submodules
├── results/                    Where your run results land
├── .github/workflows/          Automated judging on every push
├── docs/                       This guide
└── workshop/                   ROS 2 teaching material from the workshop
```

The one file you are meant to open first:
**[`race_ws/src/team_driver/team_driver/driver.py`](race_ws/src/team_driver/team_driver/driver.py)**

---

## The guide

| Chapter | What is in it |
| --- | --- |
| **[1. Setup](docs/01-setup.md)** | Installing Docker, building the image, first run, troubleshooting |
| **[2. The simulator](docs/02-simulator.md)** | Topics, message types, changing tracks, RViz, ground-truth odometry |
| **[3. ROS 2 primer](docs/03-workshop.md)** | Nodes, topics and the workshop slides, if ROS is new to you |
| **[4. Algorithms](docs/04-algorithms.md)** | The menu: reactive, planned, model-based and learned — what each needs and where each breaks |
| **[5. Evaluation](docs/05-evaluation.md)** | Scoring yourself, reading result files, how judging day runs |
| **[6. Rules](docs/06-rules.md)** | The rules, the scoring formula, and what gets you disqualified |
| **[7. Submission](docs/07-submission.md)** | What to hand in, how, and what the interview covers |

---

## Rules at a glance

The full rules are in **[docs/06-rules.md](docs/06-rules.md)** and they are what
counts. The short version:

- **Teams of 3 to 5.**
- **Deadline: 18 October 2026, 23:59 SGT.** Late entries are not scored.
- **AI assistants are allowed.** You will be asked to explain your code at the
  interview — generally, not line by line — so do not submit anything you
  cannot defend. A learned policy is held to the same standard, and meets it
  the same way: explain how you trained it and why, not what each weight
  means.
- **Do not modify the judging environment.** Check yourself with
  `./scripts/verify_judging_env.sh`.
- **Score (out of 100):**

  | | Weight | Formula |
  | --- | --- | --- |
  | Fastest single lap | 50 | `50 × (fastest lap of any team ÷ your fastest lap)` |
  | 10-lap total | 50 | `50 × (fastest 10-lap total ÷ your 10-lap total)` |

  One warm-up lap is granted before timing starts. Each collision adds 10 s to
  the lap it happened on. More than 10 collisions in a run is a
  disqualification.

- **Bonus marks** at the interview, for work you can explain properly:
  replacing the ground-truth odometry with your own localisation; generating a
  racing line from the map at runtime; or an ambitious driving algorithm —
  reinforcement learning, MPC, imitation learning.

---

## Getting help

- Check the troubleshooting section at the end of
  [docs/01-setup.md](docs/01-setup.md) first — most problems are there.
- Bring the exact error text and what you ran to the team channel.
- `./scripts/verify_judging_env.sh` will tell you if your environment has
  drifted from the official one.
- You may email `ntu-deepspeed@e.ntu.edu.sg` for further inquiries if you cannot solve the issues after troubleshooting.

Good luck. Go fast.

</details>
