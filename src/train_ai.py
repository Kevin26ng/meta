"""
Multi-Algorithm Training Pipeline for ChaosLab.

Trains CPU-based RL models on the ChaosLab SRE environment
and saves them to the models/ directory.

Models:
  - PPO (Neural Network, via stable-baselines3)
  - Q-Learning (Tabular, pure Python — NO neural network)

Usage:
    python -m src.train_ai                    # Train all models
    python -m src.train_ai --algo ppo         # Train only PPO
    python -m src.train_ai --algo qlearning   # Train only Q-Learning
    python -m src.train_ai --steps 10000      # Override training steps
"""

import argparse
import os
import time

import gymnasium as gym
import src.rl_env  # registers ChaosLab-v0


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_STEPS = 10_000
DEFAULT_QL_EPISODES = 200


def train_ppo(total_timesteps: int, scenario: str = "full_incident"):
    """Train PPO (Neural Network) model."""
    from stable_baselines3 import PPO

    print(f"\n{'='*60}")
    print(f"  TRAINING: PPO (Proximal Policy Optimization)")
    print(f"  Type: Neural Network (MLP)")
    print(f"  Steps: {total_timesteps:,}")
    print(f"  Scenario: {scenario}")
    print(f"{'='*60}\n")

    env = gym.make("ChaosLab-v0", scenario=scenario)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start_time

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "ppo_model")
    model.save(out_path)

    file_size = os.path.getsize(out_path + ".zip") / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  ✓ PPO TRAINING COMPLETE")
    print(f"  Time:  {elapsed:.1f}s")
    print(f"  Saved: {out_path}.zip ({file_size:.2f} MB)")
    print(f"{'='*60}\n")

    env.close()
    return out_path + ".zip"


def train_qlearning(total_timesteps: int, scenario: str = "full_incident"):
    """Train Tabular Q-Learning (NO neural network)."""
    from src.qlearning_agent import train_qlearning as _train_ql

    # Convert timesteps to episodes (approx 50 steps per episode)
    episodes = max(50, total_timesteps // 50)

    print(f"\n{'='*60}")
    print(f"  TRAINING: TABULAR Q-LEARNING")
    print(f"  Type: Q-Value Lookup Table (NO neural network)")
    print(f"  Episodes: {episodes}")
    print(f"  Scenario: {scenario}")
    print(f"{'='*60}\n")

    start_time = time.time()

    agent = _train_ql(
        scenario=scenario,
        episodes=episodes,
        alpha=0.15,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        verbose=True,
    )

    elapsed = time.time() - start_time

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "qlearning_model.json")
    agent.save(out_path)

    file_size = os.path.getsize(out_path) / 1024  # KB for Q-tables

    stats = agent.stats()

    print(f"\n{'='*60}")
    print(f"  ✓ Q-LEARNING TRAINING COMPLETE")
    print(f"  Time:  {elapsed:.1f}s")
    print(f"  Saved: {out_path} ({file_size:.1f} KB)")
    print(f"  Q-table entries: {stats['table_entries']:,}")
    print(f"  Unique states:   {stats['unique_states']:,}")
    print(f"{'='*60}\n")

    return out_path


ALGO_TRAINERS = {
    "ppo": train_ppo,
    "qlearning": train_qlearning,
}


def train_all(total_timesteps: int, scenario: str = "full_incident"):
    """Train all available algorithms sequentially."""
    print("\n" + "=" * 60)
    print("   CHAOSLAB MULTI-MODEL TRAINING PIPELINE")
    print(f"   Models: PPO (Neural Net) + Q-Learning (Tabular)")
    print(f"   Steps: {total_timesteps:,}")
    print("=" * 60)

    results = {}
    total_start = time.time()

    for algo, trainer in ALGO_TRAINERS.items():
        try:
            path = trainer(total_timesteps, scenario)
            results[algo] = {"status": "success", "path": path}
        except Exception as e:
            print(f"[ERROR] Failed to train {algo}: {e}")
            results[algo] = {"status": "failed", "error": str(e)}

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("   TRAINING SUMMARY")
    print("=" * 60)
    for algo, info in results.items():
        status = "✓" if info["status"] == "success" else "✗"
        detail = info.get("path", info.get("error", ""))
        print(f"   {status} {algo.upper():>12s}  →  {detail}")
    print(f"\n   Total Time: {total_elapsed:.1f}s")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChaosLab AI Training Pipeline")
    parser.add_argument("--algo", type=str, default=None,
                        choices=list(ALGO_TRAINERS.keys()),
                        help="Train a specific algorithm (default: train all)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Training timesteps (default: {DEFAULT_STEPS:,})")
    parser.add_argument("--scenario", type=str, default="full_incident",
                        help="Scenario to train on (default: full_incident)")

    args = parser.parse_args()

    if args.algo:
        ALGO_TRAINERS[args.algo](args.steps, args.scenario)
    else:
        train_all(args.steps, args.scenario)
