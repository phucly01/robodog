#!/usr/bin/env python3
# =============================================================================
# QUADRUPED-K1 // Simulation + Hardware Environment
# =============================================================================
#
# Architecture:
#   QuadrupedBase  — abstract interface (policy never touches impl)
#   QuadrupedSim   — PyBullet backend for RL training
#   QuadrupedReal  — ArduinoLink backend for K230 deployment
#
# ACTION SPACE (delta actions):
#   Policy outputs (12,) values in [-1, 1]
#   Each value is scaled by DELTA_MAX (0.2 rad) and added to current position
#   Result is clamped to per-joint limits before commanding
#   At 50Hz: ±0.2 rad/step = ±10 rad/s ≈ MG996R physical speed limit
#
# OBSERVATION SPACE (25,) float32:
#   joint_angles (12,) radians   — current joint positions
#   imu_rpy      (3,)  degrees   — roll, pitch, yaw
#   contact      (4,)  binary    — L1, R1, L2, R2 toe contact
#   body_pos     (3,)  metres    — x, y, z world position
#   body_vel     (3,)  m/s       — vx, vy, vz world velocity
#
# rex.urdf joint conventions (ALL IN RADIANS, 0 = neutral):
#   shoulder  axis=X  limits: -1.0  to +1.0   abduction (side/side)
#   leg       axis=Y  limits: -2.17 to +0.97  hip flexion (fwd/back)
#   foot      axis=Y  limits: -0.1  to +2.59  knee
#
# Install:
#   pip install pybullet numpy gymnasium stable-baselines3
# =============================================================================

from __future__ import annotations
import abc
import math
import time
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("quad_env")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
NUM_JOINTS = 12
NUM_LEGS   = 4
SIM_HZ     = 240
CTRL_HZ    = 50
SIM_STEPS_PER_CTRL = SIM_HZ // CTRL_HZ   # 4 physics steps per policy step

# Per-joint position limits (radians) from rex.urdf
ANGLE_MIN = np.array([
    -1.0, -2.17, -0.1,   # L1: shoulder, leg, foot
    -1.0, -2.17, -0.1,   # R1
    -1.0, -2.17, -0.1,   # L2
    -1.0, -2.17, -0.1,   # R2
], dtype=np.float32)

ANGLE_MAX = np.array([
    1.0, 0.97, 2.59,     # L1
    1.0, 0.97, 2.59,     # R1
    1.0, 0.97, 2.59,     # L2
    1.0, 0.97, 2.59,     # R2
], dtype=np.float32)

# Delta action scale: policy [-1,1] * DELTA_MAX = joint delta (radians)
DELTA_MAX = 0.2   # rad/step — matches MG996R physical speed at 6V

# Home pose: stable crouched stand (~0.18-0.24m body clearance)
#   shoulder = 0.0  legs directly under body
#   leg      = -0.5 thigh angled slightly forward (-29 deg)
#   foot     =  0.9 knee bent to compensate (+52 deg)
HOME_ANGLES = np.array([
    0.0, -0.5, 0.9,
    0.0, -0.5, 0.9,
    0.0, -0.5, 0.9,
    0.0, -0.5, 0.9,
], dtype=np.float32)

# Toe PyBullet link indices (confirmed from rex.urdf joint list output):
#   front_left_toe=6, front_right_toe=11, rear_left_toe=16, rear_right_toe=21
# Order matches servo map: [L1, R1, L2, R2]
TOE_LINK_IDS = [6, 11, 16, 21]

# Leg index map: leg_name -> (shoulder, leg, foot) servo channels
LEG_INDICES = {
    "L1": (0, 1, 2),
    "R1": (3, 4, 5),
    "L2": (6, 7, 8),
    "R2": (9, 10, 11),
}


# ---------------------------------------------------------------------------
# REWARD CONFIG
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    # Phase 1: learn to stand and survive
    # Once mean episode length > 500 steps, increase forward_vel to 2.0
    forward_vel:    float =  2.0    # reward forward velocity (vx)
    lateral_pen:    float = -0.5    # penalise sideways drift (vy)
    roll_pen:       float = -0.5    # penalise |roll|
    pitch_pen:      float = -0.5    # penalise |pitch|
    height_pen:     float = -5.0    # penalise height deviation
    action_smooth:  float = -0.05   # penalise large deltas
    contact_timing: float =  1.0    # reward >=3 feet on ground (high — survival first)
    alive_bonus:    float =  1.0    # per-step survival (high — survival first)
    fall_penalty:   float = -5.0    # terminal fall (lower so gradient not too harsh)

    target_height:  float = 0.20    # metres, rex crouched stand
    height_tol:     float = 0.03    # +/- metres before penalty (slightly more tolerant)
    max_roll_deg:   float = 45.0    # fall threshold (slightly more forgiving)
    max_pitch_deg:  float = 45.0


# ---------------------------------------------------------------------------
# ABSTRACT BASE CLASS
# ---------------------------------------------------------------------------
class QuadrupedBase(abc.ABC):
    """
    Gymnasium-compatible base. Policy only ever touches this class.
    Swap QuadrupedSim <-> QuadrupedReal without changing policy code.
    """

    def __init__(self, reward_weights: Optional[RewardWeights] = None):
        self.rw              = reward_weights or RewardWeights()
        self._current_angles = HOME_ANGLES.copy()   # tracks actual joint pos
        self._step_count     = 0
        self._episode_reward = 0.0

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _apply_angles(self, angles: np.ndarray) -> None:
        """Command joint positions (radians) to backend."""

    @abc.abstractmethod
    def _get_joint_angles(self) -> np.ndarray:
        """Read current joint positions (12,) radians."""

    @abc.abstractmethod
    def _get_imu_rpy(self) -> np.ndarray:
        """Read orientation [roll, pitch, yaw] degrees."""

    @abc.abstractmethod
    def _get_contact(self) -> np.ndarray:
        """Read binary toe contact (4,) [L1, R1, L2, R2]."""

    @abc.abstractmethod
    def _get_body_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read (pos metres (3,), vel m/s (3,)) world frame."""

    @abc.abstractmethod
    def _reset_backend(self) -> None:
        """Reset backend to home state."""

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None) -> tuple[dict, dict]:
        self._reset_backend()
        self._current_angles = HOME_ANGLES.copy()
        self._step_count     = 0
        self._episode_reward = 0.0
        return self._build_obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        action : (12,) delta values in [-1, 1]
                 scaled by DELTA_MAX and added to current joint positions
        """
        # Scale policy output to radians
        delta = np.clip(action, -1.0, 1.0).astype(np.float32) * DELTA_MAX

        # Apply delta to current position, clamp to joint limits
        new_angles = np.clip(
            self._current_angles + delta,
            ANGLE_MIN, ANGLE_MAX
        ).astype(np.float32)

        self._apply_angles(new_angles)
        self._current_angles = new_angles

        obs    = self._build_obs()
        reward, info = self._compute_reward(obs, delta)
        self._episode_reward += reward
        self._step_count     += 1

        roll, pitch, _ = obs["imu_rpy"]
        terminated = (
            abs(roll)  > self.rw.max_roll_deg  or
            abs(pitch) > self.rw.max_pitch_deg or
            obs["body_pos"][2] < 0.12
        )
        if terminated:
            reward += self.rw.fall_penalty

        info["step"]           = self._step_count
        info["episode_reward"] = self._episode_reward

        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        pos, vel = self._get_body_pose()
        return {
            "joint_angles": self._get_joint_angles(),
            "imu_rpy":      self._get_imu_rpy(),
            "contact":      self._get_contact(),
            "body_pos":     pos,
            "body_vel":     vel,
        }

    def obs_as_vector(self, obs: dict) -> np.ndarray:
        """Flatten obs dict to (25,) float32 vector for policy input."""
        return np.concatenate([
            obs["joint_angles"] / math.pi,        # ~[-1, 1]
            obs["imu_rpy"]      / 90.0,           # ~[-1, 1]
            obs["contact"].astype(np.float32),    # {0, 1}
            obs["body_pos"],                      # metres
            obs["body_vel"],                      # m/s
        ]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return 25

    @property
    def action_dim(self) -> int:
        return NUM_JOINTS

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: dict,
                        delta: np.ndarray) -> tuple[float, dict]:
        rw   = self.rw
        vel  = obs["body_vel"]
        pos  = obs["body_pos"]
        rpy  = obs["imu_rpy"]
        cont = obs["contact"]

        r_fwd    = rw.forward_vel   * float(vel[0])
        r_lat    = rw.lateral_pen   * abs(float(vel[1]))
        r_roll   = rw.roll_pen      * (abs(float(rpy[0])) / 90.0)
        r_pitch  = rw.pitch_pen     * (abs(float(rpy[1])) / 90.0)

        h_err    = abs(float(pos[2]) - rw.target_height) - rw.height_tol
        r_height = rw.height_pen * max(0.0, h_err)

        # Smoothness: penalise large deltas (already in radians)
        r_smooth = rw.action_smooth * float(np.abs(delta).mean())

        feet_down = int(cont.sum())
        r_contact = rw.contact_timing * (1.0 if feet_down >= 3 else -0.5)

        r_alive  = rw.alive_bonus

        total = (r_fwd + r_lat + r_roll + r_pitch +
                 r_height + r_smooth + r_contact + r_alive)

        return total, {
            "r_forward":  r_fwd,
            "r_lateral":  r_lat,
            "r_roll":     r_roll,
            "r_pitch":    r_pitch,
            "r_height":   r_height,
            "r_smooth":   r_smooth,
            "r_contact":  r_contact,
            "r_alive":    r_alive,
            "feet_down":  feet_down,
        }


# =============================================================================
# PYBULLET SIM BACKEND
# =============================================================================
class QuadrupedSim(QuadrupedBase):
    """PyBullet simulation backend for RL training."""

    URDF_PATH = "urdf/rex.urdf"

    def __init__(
        self,
        urdf_path:      str = URDF_PATH,
        gui:            bool = False,
        reward_weights: Optional[RewardWeights] = None,
    ):
        super().__init__(reward_weights)
        self._urdf_path = urdf_path
        self._gui       = gui
        self._robot_id  = None
        self._plane_id  = None
        self._joint_ids = []   # revolute joint PyBullet indices, servo order

        self._init_pybullet()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_pybullet(self):
        import pybullet as p
        import pybullet_data
        self._p = p

        self._client = p.connect(p.GUI if self._gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self._client)

        # Resolve stl/ meshes relative to urdf/ directory
        p.setAdditionalSearchPath(
            str(Path(self._urdf_path).parent.resolve()),
            physicsClientId=self._client
        )

        if self._gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8, cameraYaw=45, cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.2],
                physicsClientId=self._client
            )

        self._load_world()

    def _load_world(self):
        import pybullet_data
        p = self._p
        c = self._client

        # Absolute path for plane.urdf — avoids search path ordering issues
        plane_path = str(Path(pybullet_data.getDataPath()) / "plane.urdf")
        self._plane_id = p.loadURDF(plane_path, physicsClientId=c)

        self._robot_id = p.loadURDF(
            str(Path(self._urdf_path).resolve()),
            [0, 0, 0.35],
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=c
        )
        log.info(f"Loaded URDF: {self._urdf_path}")

        # Collect revolute joints in document order (matches servo channel map)
        self._joint_ids = [
            i for i in range(p.getNumJoints(self._robot_id, physicsClientId=c))
            if p.getJointInfo(self._robot_id, i, physicsClientId=c)[2]
            == p.JOINT_REVOLUTE
        ]
        log.info(f"Revolute joints: {len(self._joint_ids)}")

    # ------------------------------------------------------------------
    # QuadrupedBase implementation
    # ------------------------------------------------------------------

    def _apply_angles(self, angles: np.ndarray) -> None:
        p = self._p
        c = self._client
        for i, jid in enumerate(self._joint_ids[:NUM_JOINTS]):
            p.setJointMotorControl2(
                self._robot_id, jid,
                p.POSITION_CONTROL,
                targetPosition=float(angles[i]),  # radians, no conversion
                force=8.0,        # ~MG996R stall at 6V in Nm
                maxVelocity=10.0, # rad/s cap
                physicsClientId=c
            )
        for _ in range(SIM_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=c)
            if self._gui:
                time.sleep(1.0 / SIM_HZ)

    def _get_joint_angles(self) -> np.ndarray:
        return np.array([
            self._p.getJointState(
                self._robot_id, jid,
                physicsClientId=self._client)[0]
            for jid in self._joint_ids[:NUM_JOINTS]
        ], dtype=np.float32)

    def _get_imu_rpy(self) -> np.ndarray:
        _, orn = self._p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client)
        rpy = self._p.getEulerFromQuaternion(orn)
        return np.array([math.degrees(r) for r in rpy], dtype=np.float32)

    def _get_contact(self) -> np.ndarray:
        p = self._p
        c = self._client
        return np.array([
            1 if p.getContactPoints(
                bodyA=self._robot_id, bodyB=self._plane_id,
                linkIndexA=lid, physicsClientId=c)
            else 0
            for lid in TOE_LINK_IDS
        ], dtype=np.int32)

    def _get_body_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, _ = self._p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._client)
        vel, _ = self._p.getBaseVelocity(
            self._robot_id, physicsClientId=self._client)
        return np.array(pos, dtype=np.float32), np.array(vel, dtype=np.float32)

    def _reset_backend(self) -> None:
        p = self._p
        c = self._client
        p.resetBasePositionAndOrientation(
            self._robot_id, [0, 0, 0.35],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=c
        )
        p.resetBaseVelocity(
            self._robot_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=c
        )
        for i, jid in enumerate(self._joint_ids[:NUM_JOINTS]):
            p.resetJointState(
                self._robot_id, jid,
                targetValue=float(HOME_ANGLES[i]),
                targetVelocity=0.0,
                physicsClientId=c
            )
            # Hold position during settle
            p.setJointMotorControl2(
                self._robot_id, jid,
                p.POSITION_CONTROL,
                targetPosition=float(HOME_ANGLES[i]),
                force=8.0,
                physicsClientId=c
            )
        # Settle for 0.5s under gravity
        for _ in range(SIM_HZ // 2):
            p.stepSimulation(physicsClientId=c)

    def close(self):
        self._p.disconnect(self._client)


# =============================================================================
# REAL HARDWARE BACKEND (K230)
# =============================================================================
class QuadrupedReal(QuadrupedBase):
    """
    ArduinoLink-backed environment for K230 deployment.
    Requires uart_interface.py in the same directory.
    Policy actions (radians) are converted to degrees for MG996R servos.
    """

    def __init__(
        self,
        port:           str = "/dev/ttyS1",
        reward_weights: Optional[RewardWeights] = None,
        imu=None,       # initialised MPU6050 driver
    ):
        super().__init__(reward_weights)
        from uart_interface import ArduinoLink
        self._link = ArduinoLink(port=port)
        self._imu  = imu
        self._body_pos = np.array([0.0, 0.0, self.rw.target_height],
                                  dtype=np.float32)
        self._t_prev   = time.monotonic()
        self._link.start()
        log.info("QuadrupedReal ready")

    def _apply_angles(self, angles: np.ndarray) -> None:
        # Convert radians -> degrees for Arduino/MG996R
        self._link.send_angles(np.degrees(angles).tolist())
        time.sleep(1.0 / CTRL_HZ)

    def _get_joint_angles(self) -> np.ndarray:
        # No position feedback on MG996R — return last commanded (radians)
        return self._current_angles.copy()

    def _get_imu_rpy(self) -> np.ndarray:
        if self._imu is None:
            return np.zeros(3, dtype=np.float32)
        return np.array(self._imu.get_euler(), dtype=np.float32)

    def _get_contact(self) -> np.ndarray:
        return np.array(self._link.get_contact().as_list(), dtype=np.int32)

    def _get_body_pose(self) -> tuple[np.ndarray, np.ndarray]:
        rpy = self._get_imu_rpy()
        h   = self.rw.target_height * math.cos(math.radians(float(rpy[1])))
        self._body_pos[2] = h
        return self._body_pos.copy(), np.zeros(3, dtype=np.float32)

    def _reset_backend(self) -> None:
        log.info("Resetting to home pose")
        self._link.send_angles(np.degrees(HOME_ANGLES).tolist())
        self._current_angles = HOME_ANGLES.copy()
        time.sleep(1.0)

    def close(self):
        self._link.stop()


# =============================================================================
# SMOKE TEST
# =============================================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    if mode not in ("sim", "sim-gui"):
        print(f"Usage: python3 quad_env.py [sim|sim-gui]")
        raise SystemExit(1)

    env = QuadrupedSim(gui=(mode == "sim-gui"))
    obs, _ = env.reset()

    print(f"\nObs dim        : {env.obs_dim}")
    print(f"Action dim     : {env.action_dim}")
    print(f"Action space   : delta in [-1,1] * {DELTA_MAX} rad/step")
    print(f"Body height    : {obs['body_pos'][2]:.3f} m")
    print(f"Home rpy       : {obs['imu_rpy']}")
    print(f"Home contact   : {obs['contact']}")

    total_reward = 0.0
    n_episodes   = 0
    for step in range(500):
        # Small random deltas — keeps robot near standing
        action = np.random.uniform(-0.3, 0.3, NUM_JOINTS)
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            pos = obs["body_pos"]
            rpy = obs["imu_rpy"]
            print(
                f"  step={step:3d}  "
                f"z={pos[2]:.3f}m  "
                f"rpy=[{rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}]  "
                f"feet={info['feet_down']}  "
                f"rew={reward:.3f}"
            )

        if terminated:
            n_episodes += 1
            obs, _ = env.reset()

    print(f"\nTotal reward : {total_reward:.2f}")
    print(f"Episodes     : {n_episodes}")
    env.close()
    print("Done.")
