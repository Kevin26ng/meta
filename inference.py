import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.agent import LLMAgent, SystemPrompts
from src.environment import TrainingEnv


API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()
API_KEY = os.environ.get("HF_TOKEN", "").strip()

SUCCESS_SCORE_THRESHOLD = 0.8
MAX_TOTAL_REWARD = 1.0


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    payload: Dict[str, Any] = {
        "step": step,
        "action": action,
        "reward": round(reward, 3),
        "done": done,
        "error": error,
    }
    print(f"[STEP] {json.dumps(payload, ensure_ascii=True)}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    payload: Dict[str, Any] = {
        "success": success,
        "steps": steps,
        "score": round(score, 3),
        "rewards": [round(value, 3) for value in rewards],
    }
    print(f"[END] {json.dumps(payload, ensure_ascii=True)}", flush=True)


def validate_env_config() -> None:
    missing: List[str] = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not API_KEY:
        missing.append("HF_TOKEN")
    if missing:
        raise RuntimeError(
            "Missing required environment variables for inference.py: "
            + ", ".join(missing)
        )


def build_client() -> OpenAI:
    validate_env_config()
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def make_messages(initial: Dict[str, Any], history: List[str], last_reward: float) -> List[Any]:
    system_prompt = SystemPrompts.get_sys(str(initial["task_name"]))
    user_prompt = SystemPrompts.format_observation(initial)
    if history:
        user_prompt += "\n\nRECENT HISTORY:\n" + "\n".join(history)
    user_prompt += f"\n\nLast reward: {last_reward:+.2f}"

    return [
        {"role": "system", "content": system_prompt + "\n\nReturn exactly one shell command in a fenced bash block."},
        {"role": "user", "content": user_prompt},
    ]


async def run_task(task_key: str) -> None:
    env = TrainingEnv(scenario=task_key)
    initial = env.reset()
    client = build_client()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0

    log_start(task=initial["info"]["task_name"], env=task_key, model=MODEL_NAME)

    try:
        for step in range(1, initial["info"]["max_steps"] + 1):
            if env.finished:
                break

            messages = make_messages(initial, history, last_reward)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=384,
            )
            message = str(response.choices[0].message.content or "").strip()
            command = LLMAgent.extract_command(message) or message

            result = env.step(command)
            reward = result["reward"] or 0.0
            done = result["done"]
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=command, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {command!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenEnv LLM baseline.")
    parser.add_argument("--task", default="log_analysis", help="Scenario key to run")
    parser.add_argument("--all", action="store_true", help="Run all available tasks")
    args = parser.parse_args()

    task_keys = TrainingEnv.avail_tasks() if args.all else [args.task]
    for task_key in task_keys:
        await run_task(task_key)


if __name__ == "__main__":
    asyncio.run(main())
