"""
Task definitions and grading logic.

This module provides two APIs:
  1. Legacy tasks  (REGISTRY) — the original 3 hardcoded tasks, backward-compatible.
  2. Scenario tasks (ScenarioTask) — wraps the new composable scenario engine so the
     environment can treat both identically via the Objective interface.
"""

from typing import Dict, Tuple, List, Optional
from .virtual_filesystem import SystemStore
from .terminal_emulator import Shell
from .scenarios import (
    Scenario, FaultInjector, CascadeEngine, ScenarioGrader,
    load_scenario, SCENARIO_CATALOG, DIFFICULTY_MAP, list_scenarios,
)


# ======================================================================
#  BASE CLASS
# ======================================================================

class Objective:

    def __init__(self, nm: str, diff: str, desc: str):
        self.nm = nm
        self.diff = diff
        self.desc = desc
        self.lvls: Dict[int, str] = {}
        self.max_sc = 1.0

    def eval(self, fs: SystemStore, sh: Shell) -> Tuple[float, Dict]:
        raise NotImplementedError

    def guide(self) -> str:
        raise NotImplementedError


# ======================================================================
#  LEGACY TASKS (backward-compatible)
# ======================================================================

class LogSearchTask(Objective):

    def __init__(self):
        super().__init__(nm="Log Analysis", diff="easy",
                         desc="Analyze application logs to find a critical error")
        self.lvls = {1: "Log file accessed", 2: "Error 500 found"}
        self.goal = "2026-03-30T09:22:19.234Z"

    def guide(self) -> str:
        return (
            "TASK: Log Analysis (Easy)\n\n"
            "Your objective:\n"
            "Find the timestamp of the first occurrence of \"500 Internal Server Error\"\n"
            "in the application log file.\n\n"
            "Location: /var/log/app.log\n"
            "Expected format: ISO 8601 timestamp at the start of the error line\n\n"
            "Commands you might use:\n"
            "- cat /var/log/app.log\n"
            "- grep \"500\" /var/log/app.log\n"
        )

    def eval(self, fs: SystemStore, sh: Shell) -> Tuple[float, Dict]:
        meta: Dict = {
            "task": "log_analysis", "stages_completed": [],
            "target_timestamp": self.goal, "commands_run": len(sh.log),
        }
        rec = sh.history()
        if any("app.log" in cmd for cmd in rec):
            meta["stages_completed"].append(1)
        for cmd in rec:
            if "grep" in cmd and "500" in cmd:
                out, code = sh.run(cmd)
                if code == 0 and "500" in out:
                    meta["stages_completed"].append(2)
                    if self.goal in out:
                        return 1.0, meta
        return 0.0, meta


class PermFixTask(Objective):

    def __init__(self):
        super().__init__(nm="Permission Repair", diff="medium",
                         desc="Fix file permissions for a shell script")
        self.lvls = {1: "Script located",
                     2: "Permission check performed", 3: "Script made executable"}
        self.tgt = "/home/user/scripts/cleanup.sh"

    def guide(self) -> str:
        return (
            "TASK: Permission Repair (Medium)\n\n"
            "Your objective:\n"
            "Make the cleanup.sh script executable so it can be run by the system.\n\n"
            "Location: /home/user/scripts/cleanup.sh\n"
            "Current permissions: 0644 (rw-r--r--)\n"
            "Target permissions: 0755 (rwxr-xr-x)\n\n"
            "Commands you might use:\n"
            "- ls -la /home/user/scripts/cleanup.sh\n"
            "- chmod 0755 /home/user/scripts/cleanup.sh\n"
        )

    def eval(self, fs: SystemStore, sh: Shell) -> Tuple[float, Dict]:
        meta: Dict = {
            "task": "permission_repair", "stages_completed": [],
            "script_path": self.tgt, "current_permissions": None, "is_executable": False,
        }
        rec = sh.history()
        if any("cleanup.sh" in cmd for cmd in rec):
            meta["stages_completed"].append(1)
        if any("chmod" in cmd or "ls" in cmd for cmd in rec):
            meta["stages_completed"].append(2)
        ok, perms = fs.perms(self.tgt)
        if ok:
            meta["current_permissions"] = oct(perms)
            exe = bool(perms & 0o111)
            meta["is_executable"] = exe
            if exe:
                meta["stages_completed"].append(3)
                return 1.0, meta
            elif any("chmod" in cmd for cmd in rec):
                return 0.5, meta
        return 0.0, meta


class ProcessRestoreTask(Objective):

    def __init__(self):
        super().__init__(nm="Process Recovery", diff="hard",
                         desc="Diagnose and recover a failed service")
        self.lvls = {1: "Process status checked", 2: "Process verified dead",
                     3: "Process restart attempted", 4: "Process verified online"}
        self.target = "postgres"

    def guide(self) -> str:
        return (
            "TASK: Process Recovery (Hard)\n\n"
            "Your objective:\n"
            "The postgres database service has crashed. You need to:\n"
            "1. Verify that postgres is not running (dead)\n"
            "2. Restart the postgres service\n"
            "3. Verify that postgres is back online and running\n\n"
            "Commands you might use:\n"
            "- ps (list processes)\n"
            "- systemctl restart postgres\n"
            "- systemctl status postgres\n"
        )

    def eval(self, fs: SystemStore, sh: Shell) -> Tuple[float, Dict]:
        meta: Dict = {
            "task": "process_recovery", "stages_completed": [],
            "target_process": self.target, "final_status": None,
            "recovery_steps": len(sh.log),
        }
        rec = sh.history()
        if any("ps" in cmd for cmd in rec):
            meta["stages_completed"].append(1)
        for cmd in rec[:5]:
            if "ps" in cmd:
                meta["stages_completed"].append(2)
                break
        if any("systemctl restart" in cmd or "systemctl start" in cmd for cmd in rec):
            meta["stages_completed"].append(3)
        ok, pinfo = fs.svc_info(self.target)
        if ok:
            stat = pinfo.get("status", "unknown")
            meta["final_status"] = stat
            if stat == "running":
                meta["stages_completed"].append(4)
                return 1.0, meta
            elif 3 in meta["stages_completed"]:
                return 0.5, meta
            else:
                return 0.2, meta
        return 0.0, meta


# ======================================================================
#  SCENARIO TASK — wraps the composable scenario engine
# ======================================================================

class ScenarioTask(Objective):
    """Wraps a Scenario through the Objective interface for TrainingEnv."""

    def __init__(self, scenario: Scenario):
        super().__init__(nm=scenario.name, diff=scenario.difficulty, desc=scenario.description)
        self.scenario = scenario
        self._injector: Optional[FaultInjector] = None
        self._cascade: Optional[CascadeEngine] = None
        self._grader: Optional[ScenarioGrader] = None
        self._faults_applied = False

    def setup(self, fs: SystemStore, sh: Shell) -> None:
        """Inject faults and prepare the scenario. Call once after env reset."""
        self._injector = FaultInjector(fs)
        self._cascade = CascadeEngine(fs, self._injector)
        self._grader = ScenarioGrader(fs, sh)
        # apply initial faults
        for fault in self.scenario.faults:
            self._injector.inject(fault)
        # run one cascade tick (some faults trigger immediately)
        self._cascade.tick(self.scenario.cascades)
        self._faults_applied = True

    def guide(self) -> str:
        return self.scenario.guide()

    def eval(self, fs: SystemStore, sh: Shell) -> Tuple[float, Dict]:
        # ensure grader exists (lazy init if setup wasn't called)
        if self._grader is None:
            self.setup(fs, sh)
        # tick cascades every eval
        triggered = []
        if self._cascade:
            triggered = self._cascade.tick(self.scenario.cascades)
        score, meta = self._grader.evaluate(self.scenario.objectives)
        meta["task"] = self.scenario.name
        meta["difficulty"] = self.scenario.difficulty
        meta["cascades_triggered"] = triggered
        return score, meta


# ======================================================================
#  REGISTRIES
# ======================================================================

# Legacy registry (backward compat)
REGISTRY = {
    "easy": LogSearchTask(),
    "medium": PermFixTask(),
    "hard": ProcessRestoreTask(),
}


def get_task(key: str) -> Objective:
    """
    Get a task by key. Supports:
      - Legacy difficulty names: 'easy', 'medium', 'hard'
      - Scenario keys: 'cascading_db_failure', 'disk_space_crisis', etc.
      - Difficulty aliases: 'expert' maps to default expert scenario
    """
    # legacy
    if key in REGISTRY:
        return REGISTRY[key]
    # direct scenario key
    if key in SCENARIO_CATALOG:
        return ScenarioTask(load_scenario(key))
    # difficulty alias
    if key in DIFFICULTY_MAP:
        return ScenarioTask(load_scenario(DIFFICULTY_MAP[key]))
    raise ValueError(
        f"Unknown task: '{key}'. "
        f"Available: {list(REGISTRY.keys()) + list(SCENARIO_CATALOG.keys())}"
    )


def all_task_keys() -> List[str]:
    """Return all available task/scenario keys."""
    keys = list(REGISTRY.keys())
    for k in SCENARIO_CATALOG:
        if k not in keys:
            keys.append(k)
    return keys


def task_metadata(key: str) -> Dict:
    """Return metadata for a task/scenario."""
    task = get_task(key)
    return {
        "name": task.nm,
        "difficulty": task.diff,
        "description": task.desc,
        "instructions": task.guide(),
    }
