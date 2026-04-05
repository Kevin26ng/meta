"""
Tabular Q-Learning Agent for ChaosLab.

A classic reinforcement learning agent that maintains a Q-table
(Python dict) mapping (state_hash, action) → expected reward.

NO neural networks. NO GPU. Pure Python + numpy.

The observation (4096-dim ASCII vector) is hashed into a compact
state key using a combination of keyword detection and structural
fingerprinting, keeping the table size manageable.

Implements `.predict(obs)` matching the stable-baselines3 API
so it plugs into the model registry seamlessly.

Training:
    from src.qlearning_agent import QLearningAgent, train_qlearning
    agent = train_qlearning(scenario="full_incident", episodes=500)
    agent.save("models/qlearning_model.json")
"""

from __future__ import annotations

import json
import hashlib
import numpy as np
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .rl_env import ACTION_CATALOG, NUM_ACTIONS


# ======================================================================
#  STATE HASHING — compress 4096-dim observation into a table key
# ======================================================================

# Keywords to detect in the observation text for state fingerprinting
_STATE_KEYWORDS = [
    "DEAD", "dead", "running", "RUNNING",
    "error", "500", "502",
    "Failed password", "brute",
    "permission", "executable",
    "disk", "full", "90%", "95%", "100%",
    "memory", "low",
    "cron", "FAILED",
    "nginx", "postgres", "app",
    "cleanup.sh", "backup.sh",
    "Score: 0.0", "Score: 0.2", "Score: 0.5", "Score: 0.8", "Score: 1.0",
    "/var/log", "/home/user",
]


def _decode_obs(obs: np.ndarray) -> str:
    """Decode ASCII-encoded observation back to text."""
    chars = []
    for code in obs:
        c = int(code)
        if c == 0:
            break
        if 0 < c < 128:
            chars.append(chr(c))
    return "".join(chars)


def _hash_state(obs: np.ndarray) -> str:
    """
    Hash a 4096-dim ASCII observation into a compact state key.

    Strategy: detect which keywords are present in the text and
    create a binary fingerprint. This maps the huge observation
    space into a manageable number of distinct states (~2^30 max,
    but in practice much fewer).
    """
    text = _decode_obs(obs)
    text_lower = text.lower()

    # Binary fingerprint: which keywords are present?
    bits = []
    for kw in _STATE_KEYWORDS:
        bits.append("1" if kw.lower() in text_lower else "0")

    fingerprint = "".join(bits)

    # Also include a rough hash of the first 200 chars for
    # distinguishing structurally different observations
    prefix_hash = hashlib.md5(text[:200].encode()).hexdigest()[:6]

    return f"{fingerprint}_{prefix_hash}"


# ======================================================================
#  TABULAR Q-LEARNING AGENT
# ======================================================================

class QLearningAgent:
    """
    Tabular Q-Learning with epsilon-greedy exploration.

    Q-table is a Python dict: { (state_hash, action_idx): q_value }

    Hyperparameters:
        alpha:   Learning rate (how fast Q-values update)
        gamma:   Discount factor (importance of future rewards)
        epsilon: Exploration rate (probability of random action)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: (state_hash, action) → q_value
        self.q_table: Dict[str, float] = defaultdict(float)

        # Tracking
        self._prev_state: Optional[str] = None
        self._prev_action: Optional[int] = None
        self._episode_count: int = 0
        self._total_updates: int = 0

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Pick an action given observation (SB3-compatible interface).

        When deterministic=True (inference), always pick the best action.
        When deterministic=False (training), use epsilon-greedy.
        """
        state = _hash_state(obs)

        if not deterministic and random.random() < self.epsilon:
            # Explore: random action
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            # Exploit: pick action with highest Q-value
            action = self._best_action(state)

        return np.array(action), None

    def update(
        self,
        state_obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """
        Q-Learning update rule:

        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        """
        state = _hash_state(state_obs)
        next_state = _hash_state(next_obs)

        old_q = self.q_table[f"{state}_{action}"]

        if done:
            target = reward
        else:
            # max Q-value for next state across all actions
            best_next = max(
                self.q_table.get(f"{next_state}_{a}", 0.0)
                for a in range(NUM_ACTIONS)
            )
            target = reward + self.gamma * best_next

        # Q-Learning update
        self.q_table[f"{state}_{action}"] = old_q + self.alpha * (target - old_q)
        self._total_updates += 1

    def _best_action(self, state: str) -> int:
        """Return action with highest Q-value for given state."""
        best_val = float("-inf")
        best_actions = [0]

        for a in range(NUM_ACTIONS):
            q = self.q_table.get(f"{state}_{a}", 0.0)
            if q > best_val:
                best_val = q
                best_actions = [a]
            elif q == best_val:
                best_actions.append(a)

        # Break ties randomly
        return random.choice(best_actions)

    def save(self, path: str):
        """Save Q-table + metadata to JSON file."""
        data = {
            "q_table": dict(self.q_table),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "episode_count": self._episode_count,
            "total_updates": self._total_updates,
            "num_states": len(set(
                k.rsplit("_", 1)[0] for k in self.q_table.keys()
            )),
            "table_size": len(self.q_table),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load a trained Q-table from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        agent = cls(
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.95),
            epsilon=data.get("epsilon", 0.1),
        )
        agent.q_table = defaultdict(float, data["q_table"])
        agent._episode_count = data.get("episode_count", 0)
        agent._total_updates = data.get("total_updates", 0)
        return agent

    def stats(self) -> Dict[str, Any]:
        """Return training statistics."""
        unique_states = len(set(
            "_".join(k.split("_")[:-1]) for k in self.q_table.keys()
        ))
        return {
            "table_entries": len(self.q_table),
            "unique_states": unique_states,
            "episodes_trained": self._episode_count,
            "total_updates": self._total_updates,
        }

    @staticmethod
    def get_reasoning(obs: np.ndarray) -> str:
        """Return human-readable explanation of what the agent sees."""
        text = _decode_obs(obs)
        state = _hash_state(obs)
        detections = []

        checks = [
            ("dead", "⚠ Dead service"),
            ("error", "📋 Error in logs"),
            ("500", "🔴 HTTP 500"),
            ("502", "🔴 HTTP 502"),
            ("failed password", "🔒 SSH brute-force"),
            ("permission", "🔑 Permission issue"),
            ("90%", "💾 Disk >90%"),
            ("cron", "⏰ Cron issue"),
            ("memory", "🧠 Memory pressure"),
        ]

        for keyword, label in checks:
            if keyword in text.lower():
                detections.append(label)

        prefix = f"[Q-Table lookup: state={state[:20]}...]"
        if detections:
            return f"{prefix} {' | '.join(detections)}"
        return f"{prefix} 🔍 Scanning..."


# ======================================================================
#  TRAINING FUNCTION
# ======================================================================

def train_qlearning(
    scenario: str = "full_incident",
    episodes: int = 200,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    verbose: bool = True,
) -> QLearningAgent:
    """
    Train a Q-Learning agent on the ChaosLab environment.

    Uses epsilon-greedy with decay: starts exploring randomly
    and gradually shifts to exploiting learned Q-values.

    Args:
        scenario: Which scenario to train on
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Multiplicative decay per episode

    Returns:
        Trained QLearningAgent
    """
    import gymnasium as gym
    import src.rl_env  # registers ChaosLab-v0

    env = gym.make("ChaosLab-v0", scenario=scenario)
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start)

    total_rewards = []
    best_score = 0.0

    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Epsilon-greedy action selection
            action_arr, _ = agent.predict(obs, deterministic=False)
            action = int(action_arr)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-Learning update
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward

        agent._episode_count += 1
        total_rewards.append(episode_reward)

        # Decay epsilon
        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

        score = info.get("task_score", 0.0)
        if score > best_score:
            best_score = score

        if verbose and (ep + 1) % 25 == 0:
            avg_reward = sum(total_rewards[-25:]) / min(25, len(total_rewards))
            stats = agent.stats()
            print(
                f"  Episode {ep+1:>4d}/{episodes} | "
                f"Avg Reward: {avg_reward:>7.3f} | "
                f"Best Score: {best_score:.2f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Q-entries: {stats['table_entries']:,} | "
                f"States: {stats['unique_states']:,}"
            )

    env.close()

    if verbose:
        stats = agent.stats()
        print(f"\n  Training complete!")
        print(f"  Q-table: {stats['table_entries']:,} entries across {stats['unique_states']:,} states")
        print(f"  Best score achieved: {best_score:.2f}")

    return agent
