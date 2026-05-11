# robodog

A dog-style quadruped robot built from 3D-printed parts, trained with reinforcement learning to walk, navigate uneven terrain, and climb stairs. Long-term goal: transition to a humanoid configuration.

---

## Hardware

| Component | Model | Notes |
|---|---|---|
| Servos (×12) | MG996R (prototype) → Feetech STS3215 | 3 DOF per leg |
| AI Processor | CanMV K230 | Runs RL policy inference |
| Microcontroller | Arduino Mega 2560 | Drives servos via PWM |
| IMU | MPU6050 | Roll, pitch, yaw feedback |
| Toe contact | Microswitches ×4 | One per toe |
| USB adapter | FE-URT-1 | USB-to-TTL for dev/debug |
| Battery | 3S LiPo 35C | 11.1V nominal |
| Frame | PLA+ FDM | 15% gyroid infill |

### Leg Configuration (3 DOF per leg, 12 servos total)

```
Shoulder  — abduction (side to side)     axis: X
Leg       — hip flexion (forward/back)   axis: Y
Foot      — knee (up/down)               axis: Y
```

### Servo Channel Map

| Channel | Joint | Leg |
|---|---|---|
| 0 | Shoulder | Front Left (L1) |
| 1 | Leg | Front Left (L1) |
| 2 | Foot/Knee | Front Left (L1) |
| 3 | Shoulder | Front Right (R1) |
| 4 | Leg | Front Right (R1) |
| 5 | Foot/Knee | Front Right (R1) |
| 6 | Shoulder | Rear Left (L2) |
| 7 | Leg | Rear Left (L2) |
| 8 | Foot/Knee | Rear Left (L2) |
| 9 | Shoulder | Rear Right (R2) |
| 10 | Leg | Rear Right (R2) |
| 11 | Foot/Knee | Rear Right (R2) |

---

## Software Architecture

```
┌─────────────────────────────────────────────┐
│  PC (Training)                              │
│  PyBullet sim → PPO training → model.zip   │
└───────────────────┬─────────────────────────┘
                    │ export
┌───────────────────▼─────────────────────────┐
│  CanMV K230 (Inference)                     │
│  quad_env.py (QuadrupedReal)                │
│  k320_mega_interface.py → UART                   │
└───────────────────┬─────────────────────────┘
                    │ joint angles (ASCII/CSV)
┌───────────────────▼─────────────────────────┐
│  Arduino Mega 2560                          │
│  quadruped_mega.ino                         │
│  12× MG996R servos + 4× microswitches      │
└─────────────────────────────────────────────┘
```

---

## Kinematics

Local coordinate system per leg (hip as origin):

```
+Y = forward (toe swing direction)
+X = outward (abduction)
-Z = downward (stance depth)
```

### Forward Kinematics (FK)

```
toe_x = -(l2 + l3·sin(θ2)) · sin(θ1)
toe_y =  (l2 + l3·sin(θ2)) · cos(θ1)
toe_z = -l3 · cos(θ2)
```

### Inverse Kinematics (IK)

```
θ2 = acos(-z / l3)
θ1 = atan2(-x, y)
```

### Tripod Gait

```
Group A (phase 0.0): L1, L3, R2  — lift → swing → put down → stance
Group B (phase 0.5): L2, R1, R3  — offset 180° from Group A
```
Always ≥3 feet on the ground for static stability.

---

## Project Structure

```
robodog/
├── K320/
│   ├── quad_env.py          # Sim + hardware environment (QuadrupedBase, QuadrupedSim, QuadrupedReal)
│   ├── train_ppo.py         # PPO training script (Stable-Baselines3)
│   ├── k320_mega_interface.py    # K230 ↔ Arduino UART link
│   ├── setup_urdf.py        # URDF + mesh setup (run once)
│   ├── urdf/
│   │   ├── rex.urdf         # Robot description (from rex-gym, Apache 2.0)
│   │   ├── joint_map.py     # PyBullet joint index ↔ servo channel map
│   │   └── stl/             # Mesh files (from rex-gym, Apache 2.0)
│   ├── models/              # Saved checkpoints (git-ignored)
│   ├── logs/                # Training logs (git-ignored)
│   └── deps/                # Cloned dependency repos (git-ignored)
├── quadruped_mega/
│   └── quadruped_mega.ino   # Arduino Mega 2560 firmware
├── README.md
├── LICENSE
└── .gitignore
```

---

## Setup

### Prerequisites

```bash
# Clone this repo
git clone https://github.com/phucly01/robodog.git
cd robodog

# Clone rex-gym for URDF and meshes (inside K320/)
git clone https://github.com/nicrusso7/rex-gym.git K320/deps/rex-gym

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install pybullet numpy gymnasium stable-baselines3 tqdm rich pyserial
```

### URDF Setup (run once)

```bash
cd K320
python3 setup_urdf.py
```

This copies `rex.urdf` and STL meshes into `urdf/`, verifies the 12 joint channels match your servo map, and runs a PyBullet load test.

### Verify Simulation

```bash
cd K320
python3 quad_env.py sim
```

Expected output: body height ~0.24m, all 4 feet in contact at home pose, positive reward with small perturbations.

---

## Training

```bash
cd K320
# Start training (4 parallel envs, 5M steps)
python3 train_ppo.py --n-envs 4 --total-steps 5000000

# Resume from checkpoint
python3 train_ppo.py --resume models/ppo_quad_500000_steps.zip --total-steps 5000000

# Evaluate saved model
python3 train_ppo.py --eval models/ppo_quad_best/best_model.zip
```

Checkpoints are saved every 100k steps to `models/`. Best model is saved to `models/ppo_quad_best/`.

### Training Phases

| Phase | When | What to adjust |
|---|---|---|
| Phase 1: Stand | Mean episode length < 500 steps | Keep current reward weights |
| Phase 2: Walk | Episode length > 500 consistently | Increase `forward_vel` to 2.0 |
| Phase 3: Terrain | Walking stable | Add terrain randomization |

### Reward Weights (`quad_env.py → RewardWeights`)

| Parameter | Default | Effect |
|---|---|---|
| `forward_vel` | 0.5 | Reward per m/s forward |
| `lateral_pen` | -0.2 | Penalty for sideways drift |
| `roll_pen` | -0.5 | Penalty for left/right tilt |
| `pitch_pen` | -0.5 | Penalty for forward/back tilt |
| `height_pen` | -1.0 | Penalty for wrong stance height |
| `action_smooth` | -0.02 | Penalty for jerky joint movement |
| `contact_timing` | 1.0 | Reward for ≥3 feet on ground |
| `alive_bonus` | 1.0 | Flat reward for surviving each step |
| `fall_penalty` | -5.0 | One-time penalty on termination |
| `target_height` | 0.20m | Desired body clearance |
| `max_roll_deg` | 45° | Fall/reset threshold |
| `max_pitch_deg` | 45° | Fall/reset threshold |

---

## UART Protocol (K230 ↔ Arduino)

**Joint angles — K230 → Arduino (50 Hz):**
```
<90,85,70,90,85,70,90,85,70,90,85,70>\n
```
12 comma-separated degree values wrapped in `< >`.

**Toe contact — Arduino → K230 (on change only):**
```
C:1,0,1,1\n
```
4 binary values: L1, R1, L2, R2. `1` = contact, `0` = airborne.

---

## Key Lessons Learned

- Each leg has its own local coordinate system with the hip as origin
- Knee servo 90° = straight down, values below 90° = leg extends outward
- Left and right hip servos are mirrored — same physical motion requires opposite sign
- Tripod gait requires ≥3 feet planted at all times for static stability
- Gait sequence: lift → swing forward → put down → swing back → stance
- Current feedback on STS3215 doubles as a torque/contact sensor
- PPO with parallel environments outperforms SAC for sim-to-real transfer
- Curriculum learning: stand first, then walk, then handle terrain

---

## Roadmap

- [x] IK/FK foundation
- [x] PyBullet simulation environment
- [x] PPO training pipeline
- [x] Arduino firmware (MG996R + microswitches)
- [x] K230 UART interface
- [ ] RL policy: standing stable
- [ ] RL policy: forward walking
- [ ] Hardware integration test (MG996R)
- [ ] Upgrade to Feetech STS3215 (current feedback)
- [ ] Uneven terrain policy
- [ ] Stair climbing
- [ ] Humanoid transition study

---

## Attribution & Licenses

### URDF and Mesh Files
`urdf/rex.urdf` and all files in `urdf/stl/` are adapted from
[rex-gym](https://github.com/nicrusso7/rex-gym) by Nicola Russo,
licensed under the **Apache License 2.0**.

The original SpotMicro robot design is by Deok-yeon Kim,
available on [Thingiverse](https://www.thingiverse.com/thing:3761340).

### This Project
All other source files (`quad_env.py`, `train_ppo.py`, `k320_mega_interface.py`,
`setup_urdf.py`, `quadruped_mega.ino`) are original work released under the
**Apache License 2.0** — see `LICENSE` file.

---

## Dependencies

| Package | Purpose | License |
|---|---|---|
| PyBullet | Physics simulation | zlib |
| Stable-Baselines3 | PPO implementation | MIT |
| Gymnasium | RL environment interface | MIT |
| NumPy | Numerical computing | BSD |
| PySerial | UART communication | BSD |
| PyTorch | Neural network backend | BSD |
