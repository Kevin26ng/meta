import argparse
import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.agent import LLMAgent, SystemPrompts
from src.environment import TrainingEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.3").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Validate
if not MODEL_NAME:
    raise ValueError("MODEL_NAME is not set in environment variables")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN (API key) is not set in environment variables")
BENCHMARK = os.getenv("BENCHMARK_NAME", "linux-sre-env").strip()


def _safe_int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _safe_float(raw: str, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return default


MAX_STEPS_OVERRIDE = _safe_int(os.getenv("MAX_STEPS", "0"), 0)
TEMPERATURE = _safe_float(os.getenv("INFER_TEMPERATURE", "0.2"), 0.2)
MAX_TOKENS = _safe_int(os.getenv("INFER_MAX_TOKENS", "320"), 320)
SUCCESS_SCORE_THRESHOLD = _safe_float(
    os.getenv("SUCCESS_SCORE_THRESHOLD", "0.8"), 0.8)
GLOBAL_TIMEOUT = _safe_int(
    os.getenv("GLOBAL_TIMEOUT", "1140"), 1140)  # 19 min (1 min buffer)
PER_TASK_TIMEOUT = _safe_int(
    os.getenv("PER_TASK_TIMEOUT", "180"), 180)  # 3 min per task
MAX_STEPS_CAP = 15  # hard cap per task to stay within runtime budget


def validate_required_env() -> None:
    missing: List[str] = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _one_line(value: str) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = _one_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={_one_line(action)} reward={reward:.2f} done={_bool_text(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_text(success)} steps={steps} score={score:.3f} rewards={rewards_text}",
        flush=True,
    )


def build_client() -> OpenAI:
    validate_required_env()
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def build_messages(initial: Dict[str, Any], history: List[str], step: int, last_reward: float) -> List[Any]:
    system_prompt = (
        SystemPrompts.get_sys(str(initial.get("task_name", "linux-sre")))
        + "\n\nReturn exactly one shell command for the next best move."
    )

    observation = SystemPrompts.format_observation(initial)
    recent = "\n".join(history[-4:]) if history else "None"
    user_prompt = (
        f"{observation}\n\n"
        f"Step: {step}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Recent attempts:\n{recent}\n\n"
        "Now provide the next command."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def decide_action(
    client: OpenAI,
    initial: Dict[str, Any],
    history: List[str],
    step: int,
    last_reward: float,
) -> Tuple[str, Optional[str]]:
    try:
        messages: List[Any] = build_messages(
            initial, history, step, last_reward)
        response: Any = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = str(response.choices[0].message.content or "").strip()
        command = LLMAgent.extract_command(raw_text) or raw_text
        command = _one_line(command)
        if not command:
            return "echo noop", "empty_model_output"
        return command, None
    except Exception as exc:
        return "echo fallback", _one_line(str(exc))


async def run_task(task_key: str) -> None:
    env = TrainingEnv(scenario=task_key)
    reset_out = env.reset()
    prompt_state: Dict[str, Any] = {
        "task_name": reset_out["info"]["task_name"],
        "task_instructions": reset_out["info"]["instructions"],
        "observation": reset_out["observation"],
        "max_steps": reset_out["info"]["max_steps"],
    }
    client = build_client()

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0

    max_steps = MAX_STEPS_OVERRIDE if MAX_STEPS_OVERRIDE > 0 else min(
        int(reset_out["info"]["max_steps"]), MAX_STEPS_CAP)

    log_start(task=task_key, env=BENCHMARK, model=MODEL_NAME)
    task_start = time.monotonic()

    try:
        for step in range(1, max_steps + 1):
            if env.finished:
                break
            if time.monotonic() - task_start > PER_TASK_TIMEOUT:
                break

            command, model_error = decide_action(
                client, prompt_state, history, step, last_reward)
            result = env.step(command)

            reward = float(result.get("reward") or 0.0)
            done = bool(result.get("done"))
            info = result.get("info", {})
            score = float(info.get("task_score", score))
            if "observation" in result:
                prompt_state["observation"] = result["observation"]

            step_error = model_error
            if not step_error:
                raw_error = info.get("last_action_error") or info.get("error")
                if raw_error:
                    step_error = _one_line(str(raw_error))

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=command, reward=reward,
                     done=done, error=step_error)
            history.append(f"step={step} action={command} reward={reward:.2f}")

            if done:
                break

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        steps_taken += 1
        log_step(
            step=steps_taken,
            action="internal_error",
            reward=0.0,
            done=True,
            error=_one_line(str(exc)),
        )

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Custom OpenEnv inference runner.")
    parser.add_argument("--task", default="log_analysis",
                        help="Scenario key to run")
    parser.add_argument("--all", action="store_true",
                        help="Run every available task")
    args = parser.parse_args()

    validate_required_env()

    task_keys = TrainingEnv.avail_tasks() if args.all else [args.task]
    global_start = time.monotonic()
    for key in task_keys:
        if time.monotonic() - global_start > GLOBAL_TIMEOUT:
            print(
                f"[END] success=false steps=0 score=0.000 rewards= # skipped (global timeout)", flush=True)
            break
        await run_task(key)


if __name__ == "__main__":
    asyncio.run(main())
