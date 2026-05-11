#!/usr/bin/env python3
# =============================================================================
# QUADRUPED-K1 // PPO Training Script
# =============================================================================
#
# Uses Stable-Baselines3 PPO with a Gymnasium wrapper around QuadrupedSim.
# Trains a walking policy using delta actions (±0.2 rad/step).
#
# Install:
#   pip install stable-baselines3 gymnasium pybullet numpy torch
#
# Train:
#   python3 train_ppo.py
#
# Resume from checkpoint:
#   python3 train_ppo.py --resume models/ppo_quad_500000_steps.zip
#
# Evaluate saved model:
#   python3 train_ppo.py --eval models/ppo_quad_best/best_model.zip
# =============================================================================

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from quad_env import (
    QuadrupedSim,
    RewardWeights,
    NUM_JOINTS,
    ANGLE_MIN,
    ANGLE_MAX,
    HOME_ANGLES,
    DELTA_MAX,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_ppo")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
MODEL_DIR  = Path("models")
LOG_DIR    = Path("logs")
BEST_DIR   = MODEL_DIR / "ppo_quad_best"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
BEST_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# GYMNASIUM WRAPPER
# ---------------------------------------------------------------------------
class QuadrupedGymEnv(gym.Env):
    """
    Thin Gymnasium wrapper around QuadrupedSim.

    Action space  : Box(12,) in [-1, 1]  (delta fractions)
    Observation   : Box(25,) float32
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        urdf_path:      str = "urdf/rex.urdf",
        gui:            bool = False,
        reward_weights: RewardWeights = None,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        self._env = QuadrupedSim(
            urdf_path=urdf_path,
            gui=gui,
            reward_weights=reward_weights,
        )
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps     = 0

        # Action: 12 delta values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(NUM_JOINTS,),
            dtype=np.float32
        )

        # Observation: 25-dim vector
        obs_low  = np.full(self._env.obs_dim, -10.0, dtype=np.float32)
        obs_high = np.full(self._env.obs_dim,  10.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict, info = self._env.reset()
        self._elapsed_steps = 0
        return self._env.obs_as_vector(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        self._elapsed_steps += 1

        # Enforce episode length limit
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        obs_vec = self._env.obs_as_vector(obs_dict)
        return obs_vec, reward, terminated, truncated, info

    def render(self):
        pass   # GUI mode handled at QuadrupedSim level

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------
class TrainingLogger(BaseCallback):
    """Logs mean reward and episode info to console every N episodes."""

    def __init__(self, log_freq_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self._log_freq  = log_freq_episodes
        self._ep_count  = 0
        self._ep_rewards = []
        self._ep_lengths = []

    def _on_step(self) -> bool:
        # SB3 stores episode info in self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_count  += 1
                self._ep_rewards.append(info["episode"]["r"])
                self._ep_lengths.append(info["episode"]["l"])

                if self._ep_count % self._log_freq == 0:
                    mean_r = np.mean(self._ep_rewards[-self._log_freq:])
                    mean_l = np.mean(self._ep_lengths[-self._log_freq:])
                    steps  = self.num_timesteps
                    log.info(
                        f"  Episodes={self._ep_count:5d}  "
                        f"Steps={steps:8d}  "
                        f"MeanReward={mean_r:7.2f}  "
                        f"MeanLen={mean_l:5.0f}"
                    )
        return True


# ---------------------------------------------------------------------------
# TRAINING CONFIG
# ---------------------------------------------------------------------------
def make_env(rank: int, seed: int = 0):
    """Factory for vectorised environment creation."""
    def _init():
        env = QuadrupedGymEnv(
            urdf_path="urdf/rex.urdf",
            gui=False,
            max_episode_steps=1000,
        )
        env = Monitor(env, str(LOG_DIR / f"env_{rank}"))
        env.reset(seed=seed + rank)
        return env
    return _init


def build_ppo(env, tensorboard_log: str = None) -> PPO:
    """
    PPO hyperparameters tuned for legged locomotion.
    Follows roughly the ETH Zurich / SB3 quadruped recommendations.
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        # Core PPO
        n_steps=2048,          # rollout length per env per update
        batch_size=512,        # minibatch size
        n_epochs=10,           # gradient steps per update
        gamma=0.99,            # discount factor
        gae_lambda=0.95,       # GAE lambda
        clip_range=0.2,        # PPO clip epsilon
        clip_range_vf=None,    # no value function clipping
        ent_coef=0.005,        # entropy bonus (encourages exploration)
        vf_coef=0.5,           # value function loss weight
        max_grad_norm=0.5,     # gradient clipping

        # Learning rate — linear decay schedule
        learning_rate=3e-4,

        # Network architecture
        # Two separate MLPs for policy and value function
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],   # policy network layers
                vf=[256, 256],   # value network layers
            ),
            activation_fn=__import__("torch").nn.ELU,
        ),

        tensorboard_log=tensorboard_log,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def train(args):
    log.info("=" * 60)
    log.info("QUADRUPED-K1 // PPO Training")
    log.info("=" * 60)
    log.info(f"  Envs          : {args.n_envs}")
    log.info(f"  Total steps   : {args.total_steps:,}")
    log.info(f"  Save freq     : {args.save_freq:,} steps")
    log.info(f"  TensorBoard   : {args.tensorboard}")

    # Vectorised environments
    if args.n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        vec_env = make_vec_env(
            lambda: QuadrupedGymEnv(
                urdf_path="urdf/rex.urdf",
                gui=False,
                max_episode_steps=1000,
            ),
            n_envs=1,
            monitor_dir=str(LOG_DIR),
        )

    # Observation normalisation — very helpful for locomotion
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Eval env must be wrapped in VecNormalize to match training env
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([lambda: Monitor(
        QuadrupedGymEnv(urdf_path="urdf/rex.urdf", gui=False,
                        max_episode_steps=1000),
        str(LOG_DIR / "eval")
    )])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(MODEL_DIR),
        name_prefix="ppo_quad",
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(LOG_DIR),
        eval_freq=max(args.save_freq // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    logger_cb = TrainingLogger(log_freq_episodes=20)

    # Build or resume model
    tb_log = str(LOG_DIR / "tensorboard") if args.tensorboard else None

    if args.resume:
        log.info(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=vec_env,
            tensorboard_log=tb_log,
        )
    else:
        model = build_ppo(vec_env, tensorboard_log=tb_log)

    log.info(f"Policy params: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Train
    t_start = time.time()
    log.info("Training started...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_cb, eval_cb, logger_cb],
        reset_num_timesteps=not bool(args.resume),
        progress_bar=True,
    )

    # Save final model + normalisation stats
    final_path = str(MODEL_DIR / "ppo_quad_final")
    model.save(final_path)
    vec_env.save(str(MODEL_DIR / "vecnormalize_final.pkl"))

    elapsed = time.time() - t_start
    log.info(f"Training complete in {elapsed/60:.1f} min")
    log.info(f"Model saved → {final_path}.zip")
    log.info(f"Best model  → {BEST_DIR}/best_model.zip")


def evaluate(args):
    """Load a saved model and run evaluation episodes."""
    log.info(f"Evaluating: {args.eval}")

    env = QuadrupedGymEnv(
        urdf_path="urdf/rex.urdf",
        gui=args.gui,
        max_episode_steps=1000,
    )

    model = PPO.load(args.eval)

    # Load normalisation stats if available
    norm_path = str(MODEL_DIR / "vecnormalize_final.pkl")
    if Path(norm_path).exists():
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        vec_env = VecNormalize.load(
            norm_path,
            DummyVecEnv([lambda: env])
        )
        vec_env.training   = False
        vec_env.norm_reward = False
        use_vec = True
    else:
        use_vec = False

    total_rewards = []
    for ep in range(args.n_eval_episodes):
        if use_vec:
            obs = vec_env.reset()
            done = [False]
            ep_reward = 0.0
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                ep_reward += reward[0]
        else:
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                ep_reward += reward

        total_rewards.append(ep_reward)
        log.info(f"  Episode {ep+1}: reward={ep_reward:.2f}")

    log.info(f"Mean reward over {args.n_eval_episodes} episodes: "
             f"{np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

    if not use_vec:
        env.close()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUADRUPED-K1 PPO Training")

    parser.add_argument("--n-envs",    type=int,   default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--total-steps", type=int, default=5_000_000,
                        help="Total training timesteps (default: 5M)")
    parser.add_argument("--save-freq", type=int,   default=100_000,
                        help="Checkpoint save frequency in steps (default: 100k)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--resume",    type=str,   default=None,
                        help="Path to .zip checkpoint to resume from")
    parser.add_argument("--eval",      type=str,   default=None,
                        help="Path to .zip model to evaluate (skips training)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Episodes to run during evaluation")
    parser.add_argument("--gui",       action="store_true",
                        help="Show PyBullet GUI during evaluation")

    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
