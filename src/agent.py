import json
from typing import Dict, Any, Optional, List
from .environment import TrainingEnv


class AIWorker:
    """Programmatic agent interface — drive the environment step by step."""

    def __init__(self, task_difficulty: str = "medium", scenario: str = None):
        self.engine = TrainingEnv(
            difficulty=task_difficulty, scenario=scenario)
        self.recorder: List[Dict[str, Any]] = []

    def boot(self) -> Dict[str, Any]:
        reset_out = self.engine.reset()
        return {
            "status": "reset",
            "observation": reset_out["observation"],
            "task_name": reset_out["info"]["task_name"],
            "task_instructions": reset_out["info"]["instructions"],
            "max_steps": reset_out["info"]["max_steps"],
            "message": "Environment reset. Read the task instructions and begin solving."
        }

    def invoke(self, action: str, rationale: Optional[str] = None) -> Dict[str, Any]:
        step_out = self.engine.step(action)
        resp = {
            "status": "success" if step_out["info"]["exit_code"] == 0 else "failed",
            "command": action,
            "command_output": step_out["info"].get("command_output", ""),
            "reward": step_out["reward"],
            "task_score": step_out["info"]["task_score"],
            "done": step_out["done"],
            "observation": step_out["observation"],
            "step": step_out["info"]["step"],
            "max_steps": step_out["info"]["max_steps"],
            "task_metadata": step_out["info"]["task_metadata"],
        }
        self.recorder.append({
            "command": action,
            "reasoning": rationale,
            "result": resp,
        })
        if resp["done"]:
            resp["message"] = f"Task completed! Final score: {resp['task_score']:.1f}"
        else:
            resp["message"] = f"Score: {resp['task_score']:.1f} | Steps: {resp['step']}/{resp['max_steps']}"
        return resp

    def context(self) -> str:
        state = self.engine.dump()
        recent = self.recorder[-5:] if self.recorder else []
        recent_cmds = "\n".join(
            f"  [{r['command']}] -> score={r['result']['task_score']}"
            for r in recent
        ) or "No commands executed yet"
        return (
            f"LINUX SRE ENVIRONMENT STATE:\n"
            f"=== CURRENT TASK ===\n"
            f"Name: {self.engine.task.nm}\n"
            f"Difficulty: {self.engine.difficulty}\n"
            f"Description: {self.engine.task.desc}\n\n"
            f"=== PROGRESS ===\n"
            f"Task Score: {self.engine.score:.1f}\n"
            f"Steps Taken: {self.engine.step_count}\n"
            f"Max Steps: {self.engine.limit}\n\n"
            f"=== RECENT COMMANDS ===\n{recent_cmds}\n\n"
            f"=== WORKING DIRECTORY ===\n{self.engine.terminal.cwd}\n"
        )

    def report(self) -> Dict[str, Any]:
        return {
            "task_name": self.engine.task.nm,
            "difficulty": self.engine.difficulty,
            "final_score": self.engine.score,
            "steps_used": self.engine.step_count,
            "max_steps": self.engine.limit,
            "completed": self.engine.finished,
            "efficiency": round(1 - (self.engine.step_count / self.engine.limit), 2)
            if self.engine.finished else 0.0,
            "command_history": self.engine.terminal.history(),
        }


# ======================================================================
#  LLM AGENT (optional — requires litellm)
# ======================================================================

class LLMAgent:
    """
    LLM-powered agent that autonomously solves scenarios.

    Requires `litellm` package. Supports any model LiteLLM supports:
      - openai/gpt-4o, anthropic/claude-3-opus, gemini/gemini-pro, etc.

    Usage:
        agent = LLMAgent(model="openai/gpt-4o", api_key="sk-...")
        result = agent.solve("cascading_db_failure")
    """

    def __init__(self, model: str = "openai/gpt-4o", api_key: Optional[str] = None,
                 max_turns: int = 30, verbose: bool = False):
        self.model = model
        self.api_key = api_key
        self.max_turns = max_turns
        self.verbose = verbose
        self._litellm = None

    def _ensure_litellm(self):
        if self._litellm is None:
            try:
                import litellm
                self._litellm = litellm
                if self.api_key:
                    import os
                    if "openai" in self.model.lower() or "gpt" in self.model.lower():
                        os.environ["OPENAI_API_KEY"] = self.api_key
                    elif "anthropic" in self.model.lower() or "claude" in self.model.lower():
                        os.environ["ANTHROPIC_API_KEY"] = self.api_key
                    elif "gemini" in self.model.lower():
                        os.environ["GEMINI_API_KEY"] = self.api_key
            except ImportError:
                raise ImportError(
                    "litellm is required for LLMAgent. Install with: pip install litellm"
                )

    def solve(self, scenario: str = None, difficulty: str = "medium") -> Dict[str, Any]:
        """Run the agent loop: observe → think → act → repeat until done."""
        self._ensure_litellm()

        worker = AIWorker(task_difficulty=difficulty, scenario=scenario)
        initial = worker.boot()

        messages = [
            {"role": "system", "content": SystemPrompts.get_sys(
                initial.get("task_name", difficulty))},
            {"role": "user",
                "content": SystemPrompts.format_observation(initial)},
        ]

        turns = []
        for turn in range(self.max_turns):
            response = self._litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=512,
            )
            assistant_msg = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_msg})

            # extract command from response
            command = self._extract_command(assistant_msg)
            if not command:
                messages.append(
                    {"role": "user", "content": "Please provide a command to execute."})
                continue

            result = worker.invoke(command, rationale=assistant_msg)
            turns.append({"turn": turn + 1, "command": command,
                         "score": result["task_score"]})

            if self.verbose:
                print(
                    f"[Turn {turn+1}] $ {command}  → score={result['task_score']:.1f}")

            if result["done"]:
                break

            messages.append({
                "role": "user",
                "content": (
                    f"Command output:\n{result['command_output']}\n\n"
                    f"Score: {result['task_score']:.1f} | "
                    f"Step {result['step']}/{result['max_steps']}\n\n"
                    "What command should we run next?"
                ),
            })

        summary = worker.report()
        summary["turns"] = turns
        summary["model"] = self.model
        return summary

    @staticmethod
    def _extract_command(text: str) -> Optional[str]:
        """Extract a shell command from LLM response."""
        import re
        # try code block first
        m = re.search(r'```(?:bash|sh|shell)?\s*\n(.+?)\n```', text, re.DOTALL)
        if m:
            lines = m.group(1).strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line.lstrip("$ ")
        # try inline backtick
        m = re.search(r'`([^`]+)`', text)
        if m:
            cmd = m.group(1).strip()
            if " " in cmd or cmd in ("ps", "top", "df", "free", "uptime", "whoami", "hostname"):
                return cmd
        # try lines starting with $
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("$ "):
                return line[2:]
        return None


# ======================================================================
#  PROMPT TEMPLATES
# ======================================================================

class SystemPrompts:

    @staticmethod
    def get_sys(task_name: str) -> str:
        return (
            "You are a Linux System Reliability Engineer (SRE) agent. "
            "You are troubleshooting a production incident.\n\n"
            "RULES:\n"
            "- Execute ONE command at a time\n"
            "- Wrap your command in a code block: ```bash\\ncommand\\n```\n"
            "- Briefly explain your reasoning before each command\n"
            "- Work systematically: diagnose first, then fix\n"
            "- Verify your fixes after applying them\n\n"
            "Available commands: cat, grep, ls, chmod, ps, kill, systemctl, "
            "cd, tail, head, find, wc, df, du, top, free, uptime, whoami, "
            "id, hostname, date, touch, mkdir, cp, mv, rm, netstat, curl, "
            "journalctl, crontab, iptables, mount, env, export, sort, uniq, "
            "cut, tr, tee, echo, pwd, dig\n\n"
            "Pipes (|) and redirects (>, >>) are supported.\n\n"
            f"Current task: {task_name}"
        )

    @staticmethod
    def format_observation(initial: Dict[str, Any]) -> str:
        return (
            f"TASK: {initial['task_name']}\n\n"
            f"INSTRUCTIONS:\n{initial['task_instructions']}\n\n"
            f"CURRENT STATE:\n"
            f"- Directory: {initial['observation']['current_directory']}\n"
            f"- Processes:\n{initial['observation']['processes']}\n\n"
            f"You have {initial['max_steps']} steps. Begin troubleshooting."
        )


# ======================================================================
#  DEMO
# ======================================================================

def demo():
    """Quick demo showing the agent solving the medium task."""
    agent = AIWorker(task_difficulty="medium")
    initial = agent.boot()
    print(f"Task: {initial['task_name']}")
    print(f"Scripts: {initial['observation']['filesystem']}")

    commands = [
        ("ls -la /home/user/scripts/cleanup.sh", "Check current permissions"),
        ("chmod 0755 /home/user/scripts/cleanup.sh", "Make script executable"),
        ("ls -la /home/user/scripts/cleanup.sh", "Verify permissions fixed"),
    ]

    for cmd, reason in commands:
        result = agent.invoke(cmd, rationale=reason)
        print(f"\n$ {cmd}  ({reason})")
        print(f"  Output: {result['command_output'][:120]}")
        print(f"  Score: {result['task_score']:.1f} | Done: {result['done']}")
        if result["done"]:
            break

    print(f"\nFinal report: {json.dumps(agent.report(), indent=2)}")


if __name__ == "__main__":
    demo()
