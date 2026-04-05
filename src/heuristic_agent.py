"""
Heuristic Expert Agent for ChaosLab.

A hand-coded, rule-based SRE "expert" that reads the ASCII-encoded
observation, detects keywords (DEAD, error, disk, permission, etc.),
and selects the most appropriate action from the discrete ACTION_CATALOG.

Implements a `.predict(obs)` interface identical to stable-baselines3
models, so it plugs into the same model registry and server code.

No training required — pure CPU logic.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List

from .rl_env import ACTION_CATALOG, NUM_ACTIONS


# ======================================================================
#  SRE DECISION RULES
# ======================================================================
# Each rule is (keywords_to_detect, prioritized_action_sequence).
# The agent walks through rules top-to-bottom. The first rule whose
# keywords ALL appear in the observation text fires, and the agent
# picks the first action from that rule's sequence that it hasn't
# already used in this episode.

_RULES: List[Tuple[List[str], List[str]]] = [
    # ── Phase 0: Initial Triage (always runs first) ────────────────
    (
        [],  # empty = always matches on first call
        ["ps aux", "df -h", "free -m"],
    ),

    # ── Dead Postgres → restart cascade ──────────────────────────
    (
        ["postgres", "DEAD"],
        [
            "systemctl status postgres",
            "systemctl restart postgres",
            "systemctl status postgres",
        ],
    ),
    (
        ["postgres", "dead"],
        [
            "systemctl restart postgres",
            "systemctl status postgres",
        ],
    ),

    # ── Dead App Service ──────────────────────────────────────────
    (
        ["app", "DEAD"],
        [
            "systemctl status app",
            "systemctl restart app",
            "systemctl status app",
        ],
    ),
    (
        ["app", "dead"],
        [
            "systemctl restart app",
        ],
    ),

    # ── Nginx issues ──────────────────────────────────────────────
    (
        ["nginx", "DEAD"],
        [
            "systemctl restart nginx",
            "cat /etc/nginx/sites-enabled/default",
        ],
    ),
    (
        ["502", "nginx"],
        [
            "cat /var/log/nginx/error.log",
            "cat /etc/nginx/sites-enabled/default",
            'echo "server {\\n    listen 80 default_server;\\n    server_name _;\\n\\n    location / {\\n        proxy_pass http://127.0.0.1:8080;\\n        proxy_set_header Host \\$host;\\n        proxy_set_header X-Real-IP \\$remote_addr;\\n    }\\n}" > /etc/nginx/sites-enabled/default',
            "systemctl restart nginx",
        ],
    ),

    # ── Disk Space Issues ─────────────────────────────────────────
    (
        ["9[0-9]%", "/var/log"],
        [
            "du -sh /var/log/*",
            "rm /var/log/nginx/error.log",
            "rm /var/log/app.log",
            "rm /tmp/old_debug.log",
            "df -h",
        ],
    ),
    (
        ["disk", "full"],
        [
            "df -h",
            "du -sh /var/log/*",
            "rm /var/log/nginx/error.log",
            "rm /tmp/old_debug.log",
            "df -h",
        ],
    ),

    # ── Permission / chmod ────────────────────────────────────────
    (
        ["Permission", "cleanup"],
        [
            "ls -la /home/user/scripts/cleanup.sh",
            "chmod 0755 /home/user/scripts/cleanup.sh",
            "ls -la /home/user/scripts/cleanup.sh",
        ],
    ),
    (
        ["not executable", "cleanup"],
        [
            "chmod 0755 /home/user/scripts/cleanup.sh",
        ],
    ),
    (
        ["backup", "executable"],
        [
            "touch /usr/local/bin/backup.sh",
            'echo "#!/bin/bash" > /usr/local/bin/backup.sh',
            "chmod 0755 /usr/local/bin/backup.sh",
        ],
    ),

    # ── Log Analysis ──────────────────────────────────────────────
    (
        ["500", "app.log"],
        [
            'grep "500" /var/log/app.log',
        ],
    ),
    (
        ["error", "log"],
        [
            "cat /var/log/app.log",
            'grep "error" /var/log/app.log',
            'grep "500" /var/log/app.log',
        ],
    ),

    # ── Security / SSH brute force ────────────────────────────────
    (
        ["Failed password", "ssh"],
        [
            "cat /var/log/auth.log",
            'grep "Failed" /var/log/auth.log',
            'grep "10.99.99.5" /var/log/auth.log',
            'grep "Accepted" /var/log/auth.log',
        ],
    ),
    (
        ["brute", "attack"],
        [
            "cat /var/log/auth.log",
            'grep "Failed password" /var/log/auth.log',
            'grep "10.99.99.5" /var/log/auth.log',
            'grep -v "Failed" /var/log/auth.log',
        ],
    ),

    # ── Memory Issues ─────────────────────────────────────────────
    (
        ["memory", "low"],
        [
            "free -m",
            "top",
            "ps aux",
            "systemctl restart app",
        ],
    ),

    # ── Cron Issues ───────────────────────────────────────────────
    (
        ["cron", "fail"],
        [
            "cat /var/log/cron.log",
            "crontab -l",
            "touch /usr/local/bin/backup.sh",
            'echo "#!/bin/bash" > /usr/local/bin/backup.sh',
            "chmod 0755 /usr/local/bin/backup.sh",
        ],
    ),

    # ── Network Issues ────────────────────────────────────────────
    (
        ["unreachable", "connection"],
        [
            "netstat -tlnp",
            "curl localhost:8080",
            "systemctl restart app",
            "curl localhost:8080",
        ],
    ),

    # ── Generic Fallback: more diagnostics ────────────────────────
    (
        ["error"],
        [
            "cat /var/log/app.log",
            "cat /var/log/nginx/error.log",
            "cat /var/log/auth.log",
            "ps aux",
        ],
    ),
]


def _decode_obs(obs: np.ndarray) -> str:
    """Decode a Box observation back into a text string."""
    chars = []
    for code in obs:
        c = int(code)
        if c == 0:
            break
        if 0 < c < 128:
            chars.append(chr(c))
    return "".join(chars)


def _action_index(command: str) -> Optional[int]:
    """Find the index of a command in ACTION_CATALOG, or None."""
    try:
        return ACTION_CATALOG.index(command)
    except ValueError:
        return None


class HeuristicAgent:
    """
    Rule-based SRE expert.

    Matches keyword patterns against the decoded observation text and
    walks through a prioritized action list per matched rule.

    Implements `.predict(obs, deterministic=True)` matching the SB3
    interface so it can be used as a drop-in replacement.
    """

    def __init__(self):
        self._used_actions: set = set()
        self._triage_done: bool = False
        self._step: int = 0

    def reset(self):
        self._used_actions = set()
        self._triage_done = False
        self._step = 0

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Pick an action given the current observation.

        Returns (action_array, None) matching the SB3 `.predict()` API.
        """
        self._step += 1
        text = _decode_obs(obs)

        # Phase 0: initial triage (first 3 steps)
        if not self._triage_done:
            triage_cmds = ["ps aux", "df -h", "free -m"]
            for cmd in triage_cmds:
                idx = _action_index(cmd)
                if idx is not None and idx not in self._used_actions:
                    self._used_actions.add(idx)
                    if len(self._used_actions) >= 3:
                        self._triage_done = True
                    return np.array(idx), None
            self._triage_done = True

        # Walk through rules, match keywords against observation text
        for keywords, action_sequence in _RULES:
            if not keywords:
                continue  # skip empty-rule (triage already handled)

            if all(kw.lower() in text.lower() for kw in keywords):
                # Found a matching rule — pick first unused action
                for cmd in action_sequence:
                    idx = _action_index(cmd)
                    if idx is not None and idx not in self._used_actions:
                        self._used_actions.add(idx)
                        return np.array(idx), None

        # Fallback: try generic helpful commands we haven't run yet
        fallback_cmds = [
            "ps aux", "cat /var/log/app.log", "df -h",
            "cat /var/log/auth.log", "systemctl status postgres",
            "systemctl restart postgres", "systemctl restart app",
            "systemctl restart nginx", "netstat -tlnp",
            "cat /var/log/nginx/error.log", "free -m",
            "du -sh /var/log/*", "ls -la /home/user/scripts/",
            "crontab -l", "cat /var/log/cron.log",
        ]
        for cmd in fallback_cmds:
            idx = _action_index(cmd)
            if idx is not None and idx not in self._used_actions:
                self._used_actions.add(idx)
                return np.array(idx), None

        # Absolute fallback: any unused action
        for i in range(NUM_ACTIONS):
            if i not in self._used_actions:
                self._used_actions.add(i)
                return np.array(i), None

        # All actions exhausted (shouldn't happen with 88 actions)
        return np.array(0), None

    @staticmethod
    def get_reasoning(obs: np.ndarray) -> str:
        """Return a human-readable explanation of what the agent sees."""
        text = _decode_obs(obs)
        detections = []

        checks = [
            ("DEAD", "⚠ Dead service detected"),
            ("dead", "⚠ Dead service detected"),
            ("error", "📋 Error entries in logs"),
            ("500", "🔴 HTTP 500 errors found"),
            ("502", "🔴 HTTP 502 Bad Gateway"),
            ("Failed password", "🔒 SSH brute-force activity"),
            ("9[0-9]%", "💾 Disk usage critical (>90%)"),
            ("Permission", "🔑 Permission issue detected"),
            ("cron", "⏰ Cron job issue"),
            ("memory", "🧠 Memory pressure"),
        ]

        for keyword, label in checks:
            if keyword.lower() in text.lower():
                detections.append(label)

        if not detections:
            return "🔍 Scanning environment for anomalies..."

        return " | ".join(detections)
