<<<<<<< HEAD
from typing import Dict, Tuple, Optional, List, Any
import json
from .virtual_filesystem import SystemStore
from .terminal_emulator import Shell
from .tasks import REGISTRY, Objective


class TrainingEnv:
    
    def __init__(self, difficulty: str = "medium"):
        if difficulty not in REGISTRY:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        self.difficulty = difficulty
        self.task = REGISTRY[difficulty]
=======
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
>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
        self.storage = SystemStore()
        self.terminal = Shell(self.storage)
        self.step_count = 0
        self.limit = 50
        self.finished = False
        self.score = 0.0
<<<<<<< HEAD
    
=======
        self._scenario_key = key

        # scenario tasks may override max_steps
        if isinstance(self.task, ScenarioTask):
            self.limit = self.task.scenario.max_steps

>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
    def reset(self) -> Dict[str, Any]:
        self.storage.clear()
        self.terminal = Shell(self.storage)
        self.step_count = 0
        self.finished = False
        self.score = 0.0
<<<<<<< HEAD
        
        view = self._view()
        
=======

        # for scenario tasks, inject faults into the fresh environment
        if isinstance(self.task, ScenarioTask):
            self.task.setup(self.storage, self.terminal)

        view = self._view()
>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
        return {
            "observation": view,
            "info": {
                "task_name": self.task.nm,
                "difficulty": self.difficulty,
                "instructions": self.task.guide(),
                "max_steps": self.limit,
<<<<<<< HEAD
            }
        }
    
=======
                "scenario_key": self._scenario_key,
            }
        }

>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
    def step(self, cmd: str) -> Dict[str, Any]:
        if self.finished:
            return {
                "observation": self._view(),
                "reward": 0.0,
                "done": True,
                "info": {"error": "Episode already complete. Call reset() first."}
            }
<<<<<<< HEAD
        
        self.step_count += 1
        
        out, code = self.terminal.run(cmd)
        
        penalty = -0.01
        
        done = self.step_count >= self.limit
        
        curr_score, task_info = self.task.eval(self.storage, self.terminal)
        
=======

        self.step_count += 1

        out, code = self.terminal.run(cmd)

        penalty = -0.01
        done = self.step_count >= self.limit

        curr_score, task_info = self.task.eval(self.storage, self.terminal)

>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
        if curr_score > self.score:
            bonus = (curr_score - self.score) * 0.5
            penalty += bonus
            self.score = curr_score
<<<<<<< HEAD
        
=======

>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
        if curr_score >= 1.0 and not done:
            penalty += 0.5
            done = True
            self.finished = True
<<<<<<< HEAD
        
        view = self._view()
        
=======

        if done and not self.finished:
            self.finished = True

        view = self._view()

>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
        return {
            "observation": view,
            "reward": penalty,
            "done": done,
            "info": {
                "task_score": self.score,
                "command": cmd,
<<<<<<< HEAD
=======
                "command_output": out,
>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
                "exit_code": code,
                "output_length": len(out),
                "step": self.step_count,
                "max_steps": self.limit,
                "task_metadata": task_info,
            }
        }
<<<<<<< HEAD
    
    def _view(self) -> Dict[str, Any]:
        pout, _ = self.terminal.run("ps")
        lout, _ = self.terminal.run("ls /home/user/scripts")
        
=======

    def _view(self) -> Dict[str, Any]:
        pout, _ = self.terminal.run("ps")
        lout, _ = self.terminal.run("ls /home/user/scripts")

>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
        return {
            "current_directory": self.terminal.cwd,
            "processes": pout,
            "filesystem": lout,
            "task_name": self.task.nm,
            "task_description": self.task.desc,
            "request": "Use commands to complete the task. Type your command below."
        }
<<<<<<< HEAD
    
    def dump(self) -> Dict:
        return {
            "filesystem": self.storage.snapshot(),
=======

    def dump(self) -> Dict:
        snap = self.storage.snapshot()
        return {
            "filesystem": snap,
>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
            "episode_step": self.step_count,
            "max_steps": self.limit,
            "task_score": self.score,
            "task_difficulty": self.difficulty,
<<<<<<< HEAD
            "command_history": self.terminal.history(),
        }
    
    @staticmethod
    def avail_tasks() -> List[str]:
        return list(REGISTRY.keys())
    
    @staticmethod
    def task_details(difficulty: str) -> Dict[str, Any]:
        if difficulty not in REGISTRY:
            return {}
        
        t = REGISTRY[difficulty]
        return {
            "name": t.nm,
            "difficulty": t.diff,
            "description": t.desc,
            "instructions": t.guide(),
        }
=======
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
>>>>>>> 69d7d04 (Enchancement in Environment for real world transition)
