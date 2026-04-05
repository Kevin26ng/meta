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
    load_scenario, SCENARIO_CATALOG, list_scenarios,
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


def get_task(key: str) -> Objective:
    """
    Get a task by its scenario key.
    """
    if key in SCENARIO_CATALOG:
        return ScenarioTask(load_scenario(key))
    raise ValueError(
        f"Unknown scenario key: '{key}'. "
        f"Available: {list(SCENARIO_CATALOG.keys())}"
    )


def all_task_keys() -> List[str]:
    """Return all available scenario keys."""
    return list(SCENARIO_CATALOG.keys())


def task_metadata(key: str) -> Dict:
    """Return metadata for a task/scenario."""
    task = get_task(key)
    return {
        "name": task.nm,
        "difficulty": task.diff,
        "description": task.desc,
        "instructions": task.guide(),
    }
