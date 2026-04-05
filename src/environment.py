from typing import Dict, List, Any
from .virtual_filesystem import SystemStore
from .terminal_emulator import Shell
from .tasks import (
    REGISTRY, Objective, ScenarioTask,
    get_task, all_task_keys, task_metadata,
)
from .scenarios import list_scenarios


class TrainingEnv:

    def __init__(self, difficulty: str = "medium", scenario: str = None):
        """
        Create environment. Pass either:
          - difficulty='easy'|'medium'|'hard'  (legacy tasks)
          - scenario='cascading_db_failure'    (new scenario engine)
        """
        key = scenario or difficulty
        self.task = get_task(key)
        self.difficulty = self.task.diff
        self.storage = SystemStore()
        self.terminal = Shell(self.storage)
        self.step_count = 0
        self.limit = 50
        self.finished = False
        self.score = 0.0
        self._scenario_key = key

        # scenario tasks may override max_steps
        if isinstance(self.task, ScenarioTask):
            self.limit = self.task.scenario.max_steps

    def reset(self) -> Dict[str, Any]:
        self.storage.clear()
        self.terminal = Shell(self.storage)
        self.step_count = 0
        self.finished = False
        self.score = 0.0

        # for scenario tasks, inject faults into the fresh environment
        if isinstance(self.task, ScenarioTask):
            self.task.setup(self.storage, self.terminal)

        view = self._view()
        return {
            "observation": view,
            "info": {
                "task_name": self.task.nm,
                "difficulty": self.difficulty,
                "instructions": self.task.guide(),
                "max_steps": self.limit,
                "scenario_key": self._scenario_key,
            }
        }

    def step(self, cmd: str) -> Dict[str, Any]:
        if self.finished:
            return {
                "observation": self._view(),
                "reward": 0.0,
                "done": True,
                "info": {"error": "Episode already complete. Call reset() first."}
            }

        self.step_count += 1

        out, code = self.terminal.run(cmd)

        penalty = -0.01
        done = self.step_count >= self.limit

        curr_score, task_info = self.task.eval(self.storage, self.terminal)

        if curr_score > self.score:
            bonus = (curr_score - self.score) * 0.5
            penalty += bonus
            self.score = curr_score
        if curr_score >= 1.0 and not done:
            penalty += 0.5
            done = True
            self.finished = True

        if done and not self.finished:
            self.finished = True

        view = self._view()

        return {
            "observation": view,
            "reward": penalty,
            "done": done,
            "info": {
                "task_score": self.score,
                "command": cmd,
                "command_output": out,
                "exit_code": code,
                "output_length": len(out),
                "step": self.step_count,
                "max_steps": self.limit,
                "task_metadata": task_info,
            }
        }

    def _view(self) -> Dict[str, Any]:
        pout, _ = self.terminal.run("ps")
        lout, _ = self.terminal.run("ls /home/user/scripts")

        return {
            "current_directory": self.terminal.cwd,
            "processes": pout,
            "filesystem": lout,
            "task_name": self.task.nm,
            "task_description": self.task.desc,
            "request": "Use commands to complete the task. Type your command below."
        }

    def dump(self) -> Dict:
        snap = self.storage.snapshot()
        return {
            "filesystem": snap,
            "episode_step": self.step_count,
            "max_steps": self.limit,
            "task_score": self.score,
            "task_difficulty": self.difficulty,
            "scenario_key": self._scenario_key,
            "command_history": self.terminal.history(),
        }

    # ------------------------------------------------------------------
    #  Class-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def avail_tasks() -> List[str]:
        return all_task_keys()

    @staticmethod
    def task_details(key: str) -> Dict[str, Any]:
        try:
            return task_metadata(key)
        except ValueError:
            return {}

    @staticmethod
    def avail_scenarios() -> Dict[str, Dict]:
        return list_scenarios()
