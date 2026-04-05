"""
Gymnasium-compatible RL Environment for ChaosLab.

Wraps the TrainingEnv into a standard gymnasium.Env so it can be used
with any RL library (stable-baselines3, ray/rllib, cleanrl, etc.).

The action space is DISCRETE: the agent picks from a curated list of
bash commands that are relevant to SRE troubleshooting scenarios.

The observation space is a text string (the terminal output), encoded
as a Box of ASCII character codes for compatibility with standard RL.

Usage:
    from src.rl_env import ChaosLabEnv

    env = ChaosLabEnv(scenario="cascading_db_failure")
    obs, info = env.reset()
    action = env.action_space.sample()  # random command
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple

from .environment import TrainingEnv


# ======================================================================
#  DISCRETE ACTION CATALOG
# ======================================================================
# These are the bash commands an RL agent can choose from.
# They cover every scenario in the project: diagnostics, service mgmt,
# file operations, disk/memory/network troubleshooting, and security.

ACTION_CATALOG: List[str] = [
    # ── Diagnostics & Observation ──────────────────────────────────
    "ps aux",
    "ps",
    "top",
    "df -h",
    "du -sh /var/log/*",
    "du -sh /tmp/*",
    "free -m",
    "uptime",
    "whoami",
    "hostname",
    "date",
    "pwd",

    # ── Log Investigation ──────────────────────────────────────────
    "cat /var/log/app.log",
    "cat /var/log/auth.log",
    "cat /var/log/nginx/error.log",
    "cat /var/log/nginx/access.log",
    "cat /var/log/cron.log",
    "cat /var/log/syslog",
    "tail -n 50 /var/log/app.log",
    "tail -n 50 /var/log/auth.log",
    "tail -n 50 /var/log/nginx/error.log",
    "grep \"500\" /var/log/app.log",
    "grep \"error\" /var/log/app.log",
    "grep \"Failed\" /var/log/auth.log",
    "grep \"Failed password\" /var/log/auth.log",
    "grep -i \"error\" /var/log/nginx/error.log",
    "grep \"10.99.99.5\" /var/log/auth.log",
    "grep \"Accepted\" /var/log/auth.log",
    "grep -v \"Failed\" /var/log/auth.log",

    # ── Service Management ─────────────────────────────────────────
    "systemctl status postgres",
    "systemctl restart postgres",
    "systemctl start postgres",
    "systemctl stop postgres",
    "systemctl status nginx",
    "systemctl restart nginx",
    "systemctl start nginx",
    "systemctl status app",
    "systemctl restart app",
    "systemctl start app",
    "systemctl status sshd",
    "systemctl restart sshd",

    # ── File & Permission Operations ───────────────────────────────
    "ls -la /home/user/scripts/",
    "ls -la /home/user/scripts/cleanup.sh",
    "ls -la /etc/nginx/sites-enabled/default",
    "ls -la /usr/local/bin/backup.sh",
    "chmod 0755 /home/user/scripts/cleanup.sh",
    "chmod +x /home/user/scripts/cleanup.sh",
    "chmod 0755 /usr/local/bin/backup.sh",
    "chmod +x /usr/local/bin/backup.sh",
    "cat /home/user/scripts/cleanup.sh",
    "cat /etc/nginx/sites-enabled/default",
    "touch /usr/local/bin/backup.sh",
    'echo "#!/bin/bash" > /usr/local/bin/backup.sh',

    # ── Config Repair ──────────────────────────────────────────────
    'echo "server {\n    listen 80 default_server;\n    server_name _;\n\n    location / {\n        proxy_pass http://127.0.0.1:8080;\n        proxy_set_header Host \\$host;\n        proxy_set_header X-Real-IP \\$remote_addr;\n    }\n}" > /etc/nginx/sites-enabled/default',

    # ── Disk Cleanup ───────────────────────────────────────────────
    "rm /var/log/nginx/error.log",
    "rm /var/log/app.log",
    "rm /tmp/old_debug.log",
    "find /var/log -name '*.log' -type f",
    "echo '' > /var/log/nginx/error.log",
    "echo '' > /var/log/app.log",

    # ── Network Troubleshooting ────────────────────────────────────
    "netstat -tlnp",
    "ss -tlnp",
    "curl localhost:8080",
    "curl localhost:80",
    "curl http://127.0.0.1:8080",
    "iptables -L",

    # ── Cron ───────────────────────────────────────────────────────
    "crontab -l",
    "journalctl -u cron",

    # ── Memory ─────────────────────────────────────────────────────
    "cat /proc/meminfo",

    # ── Environment ────────────────────────────────────────────────
    "env",
    "history",

    # ── Navigation ─────────────────────────────────────────────────
    "ls /",
    "ls /var/log",
    "ls /etc",
    "ls /tmp",
    "cd /var/log",
    "cd /home/user",
    "cd /",
]

# Total number of actions
NUM_ACTIONS = len(ACTION_CATALOG)

# Maximum observation length in characters
MAX_OBS_LENGTH = 4096


# ======================================================================
#  GYMNASIUM ENVIRONMENT
# ======================================================================

class ChaosLabEnv(gym.Env):
    """
    Gymnasium RL Environment for ChaosLab SRE scenarios.

    Action Space:  Discrete(N) — index into ACTION_CATALOG
    Observation:   Box(0, 127, shape=(MAX_OBS_LENGTH,)) — ASCII-encoded terminal output
    Reward:        Float from TrainingEnv (penalty per step + bonus for progress)

    Args:
        scenario: Scenario key (e.g. 'cascading_db_failure', 'disk_space_crisis')
        max_obs_len: Maximum observation string length (truncated/padded to this)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        scenario: str = "log_analysis",
        max_obs_len: int = MAX_OBS_LENGTH,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.scenario = scenario
        self.max_obs_len = max_obs_len
        self.render_mode = render_mode

        # Gymnasium spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=127,
            shape=(self.max_obs_len,),
            dtype=np.int32,
        )

        # Internal state
        self._env: Optional[TrainingEnv] = None
        self._last_output: str = ""
        self._episode_reward: float = 0.0
        self._command_history: List[str] = []

    # ── Core Gym API ───────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation."""
        super().reset(seed=seed)

        # Allow overriding scenario via options
        scenario = self.scenario
        if options:
            scenario = options.get("scenario", scenario)

        self._env = TrainingEnv(scenario=scenario)
        reset_result = self._env.reset()

        self._episode_reward = 0.0
        self._command_history = []

        # Build initial observation string
        obs_text = self._format_observation(reset_result)
        self._last_output = obs_text

        info = {
            "task_name": reset_result["info"]["task_name"],
            "instructions": reset_result["info"]["instructions"],
            "max_steps": reset_result["info"]["max_steps"],
            "scenario_key": reset_result["info"].get("scenario_key", ""),
            "action_catalog_size": NUM_ACTIONS,
        }

        return self._encode_obs(obs_text), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute a discrete action (index into ACTION_CATALOG)."""
        assert self._env is not None, "Call reset() before step()"
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        command = ACTION_CATALOG[action]
        self._command_history.append(command)

        result = self._env.step(command)

        reward = result["reward"]
        self._episode_reward += reward

        terminated = result["done"] and self._env.score >= 1.0
        truncated = result["done"] and not terminated

        # Build observation
        obs_text = (
            f"$ {command}\n"
            f"{result['info'].get('command_output', '')}\n"
            f"---\n"
            f"Score: {result['info']['task_score']:.2f} | "
            f"Step: {result['info']['step']}/{result['info']['max_steps']}"
        )
        self._last_output = obs_text

        info = {
            "command": command,
            "command_output": result["info"].get("command_output", ""),
            "task_score": result["info"]["task_score"],
            "exit_code": result["info"]["exit_code"],
            "step": result["info"]["step"],
            "max_steps": result["info"]["max_steps"],
            "episode_reward": self._episode_reward,
            "task_metadata": result["info"].get("task_metadata", {}),
        }

        if self.render_mode == "human":
            self.render()

        return self._encode_obs(obs_text), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the last command output."""
        if self.render_mode == "ansi":
            return self._last_output
        elif self.render_mode == "human":
            print(self._last_output)
        return None

    def close(self):
        """Clean up."""
        self._env = None
        
    def hijack_env(self, active_env: TrainingEnv):
        """Hijack an actively running TrainingEnv (from websockets or arena)"""
        self._env = active_env
        self._episode_reward = 0.0
        
    def get_current_obs(self) -> np.ndarray:
        """Grabs the current state for predicting actions on a hijacked env"""
        if not self._env:
            return np.zeros(self.max_obs_len, dtype=np.int32)
            
        view = self._env._view()
        
        obs_text = (
            f"=== SCENARIO: {self._env.task.nm} ===\n"
            f"Instructions:\n{self._env.task.guide()}\n\n"
            f"Current Directory: {view['current_directory']}\n"
            f"Processes:\n{view['processes']}\n"
            f"Files:\n{view['filesystem']}\n"
            f"Max Steps: {self._env.limit}\n"
        )
        return self._encode_obs(obs_text)

    # ── Helpers ────────────────────────────────────────────────────

    def _encode_obs(self, text: str) -> np.ndarray:
        """Encode a text string as a fixed-length array of ASCII codes."""
        # Truncate or pad to max_obs_len
        text = text[:self.max_obs_len]
        codes = [ord(c) if ord(c) < 128 else 32 for c in text]
        # Pad with zeros
        codes += [0] * (self.max_obs_len - len(codes))
        return np.array(codes, dtype=np.int32)

    @staticmethod
    def _format_observation(reset_result: Dict) -> str:
        """Format the initial reset observation as a readable string."""
        obs = reset_result["observation"]
        info = reset_result["info"]
        return (
            f"=== SCENARIO: {info['task_name']} ===\n"
            f"Instructions:\n{info['instructions']}\n\n"
            f"Current Directory: {obs['current_directory']}\n"
            f"Processes:\n{obs['processes']}\n"
            f"Files:\n{obs['filesystem']}\n"
            f"Max Steps: {info['max_steps']}\n"
        )

    # ── Utility Methods ────────────────────────────────────────────

    @staticmethod
    def get_action_catalog() -> List[str]:
        """Return the full list of available commands."""
        return ACTION_CATALOG.copy()

    @staticmethod
    def action_to_command(action: int) -> str:
        """Convert a discrete action index to the corresponding bash command."""
        return ACTION_CATALOG[action]

    @staticmethod
    def command_to_action(command: str) -> Optional[int]:
        """Find the action index for a given command string, or None."""
        try:
            return ACTION_CATALOG.index(command)
        except ValueError:
            return None

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get a summary of the current episode."""
        if self._env is None:
            return {}
        return {
            "task_name": self._env.task.nm,
            "difficulty": self._env.difficulty,
            "final_score": self._env.score,
            "steps_used": self._env.step_count,
            "max_steps": self._env.limit,
            "completed": self._env.finished,
            "episode_reward": self._episode_reward,
            "commands_executed": self._command_history.copy(),
            "efficiency": round(1 - (self._env.step_count / self._env.limit), 2)
            if self._env.finished else 0.0,
        }


# ======================================================================
#  REGISTER WITH GYMNASIUM
# ======================================================================

# Register so users can do: gym.make("ChaosLab-v0", scenario="...")
gym.register(
    id="ChaosLab-v0",
    entry_point="src.rl_env:ChaosLabEnv",
    kwargs={"scenario": "log_analysis"},
)
