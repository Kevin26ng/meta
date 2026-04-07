"""
Multi-Algorithm Training Pipeline for ChaosLab.

Trains CPU-based RL models on the ChaosLab SRE environment
and saves them to the models/ directory.

Models:
  - PPO (Neural Network, via stable-baselines3)
  - Q-Learning (Tabular, pure Python — NO neural network)

Usage:
    python -m src.train_ai                    # Train all models
    python -m src.train_ai --algo ppo         # Train only PPO
    python -m src.train_ai --algo qlearning   # Train only Q-Learning
    python -m src.train_ai --steps 10000      # Override training steps
    python -m src.train_ai --llm-guided       # LLM-tuned RL training plan
    python -m src.train_ai --llm-guided --llm-autorl-trials 5
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
from openai import OpenAI
import src.rl_env  # registers ChaosLab-v0


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_STEPS = 10_000
DEFAULT_QL_EPISODES = 200
DEFAULT_AUTORL_TRIALS = 1

ALLOWED_SCENARIOS = {
    "log_analysis",
    "permission_repair",
    "process_recovery",
    "cascading_db_failure",
    "disk_space_crisis",
    "cron_job_failure",
    "nginx_misconfiguration",
    "security_incident",
    "memory_leak",
    "network_troubleshooting",
    "full_incident",
}

DEFAULT_EVAL_SCENARIOS = [
    "log_analysis",
    "permission_repair",
    "process_recovery",
    "full_incident",
]


def _build_openai_client() -> OpenAI:
    """Build an OpenAI client from hackathon-required environment variables."""
    api_key = os.environ.get("HF_TOKEN", "").strip()
    base_url = os.environ.get("API_BASE_URL", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    missing = []
    if not base_url:
        missing.append("API_BASE_URL")
    if not model_name:
        missing.append("MODEL_NAME")
    if not api_key:
        missing.append("HF_TOKEN")
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON payload from model output with fenced-block tolerance."""
    text = text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start:end + 1])


def _sanitize_scenario(scenario: str, fallback: str) -> str:
    if scenario in ALLOWED_SCENARIOS:
        return scenario
    return fallback


def _sanitize_training_plan(plan: Dict[str, Any], default_scenario: str) -> Dict[str, Any]:
    """Clamp and sanitize LLM plan into safe training bounds."""
    fallback: Dict[str, Any] = {
        "source": "fallback",
        "ppo": {
            "scenario": default_scenario,
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
        "qlearning": {
            "scenario": default_scenario,
            "alpha": 0.15,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.99,
        },
    }

    ppo = plan.get("ppo", {})
    ql = plan.get("qlearning", {})
    source = str(plan.get("source", "llm"))

    return {
        "source": source,
        "ppo": {
            "scenario": _sanitize_scenario(str(ppo.get("scenario", fallback["ppo"]["scenario"])), default_scenario),
            "learning_rate": float(min(max(float(ppo.get("learning_rate", 3e-4)), 1e-5), 5e-3)),
            "n_steps": int(min(max(int(ppo.get("n_steps", 256)), 64), 1024)),
            "batch_size": int(min(max(int(ppo.get("batch_size", 64)), 16), 256)),
            "n_epochs": int(min(max(int(ppo.get("n_epochs", 10)), 3), 20)),
            "gamma": float(min(max(float(ppo.get("gamma", 0.99)), 0.9), 0.999)),
            "ent_coef": float(min(max(float(ppo.get("ent_coef", 0.01)), 0.0), 0.05)),
        },
        "qlearning": {
            "scenario": _sanitize_scenario(str(ql.get("scenario", fallback["qlearning"]["scenario"])), default_scenario),
            "alpha": float(min(max(float(ql.get("alpha", 0.15)), 0.01), 0.5)),
            "gamma": float(min(max(float(ql.get("gamma", 0.95)), 0.8), 0.999)),
            "epsilon_start": float(min(max(float(ql.get("epsilon_start", 1.0)), 0.2), 1.0)),
            "epsilon_end": float(min(max(float(ql.get("epsilon_end", 0.05)), 0.01), 0.2)),
            "epsilon_decay": float(min(max(float(ql.get("epsilon_decay", 0.99)), 0.95), 0.9995)),
        },
    }


def get_llm_training_plan(default_scenario: str, total_timesteps: int) -> Dict[str, Any]:
    """Request an RL training plan from the LLM and sanitize it."""
    fallback: Dict[str, Any] = _sanitize_training_plan({"source": "fallback"}, default_scenario)

    try:
        client = _build_openai_client()
        model_name = os.environ.get("MODEL_NAME", "").strip()
        if not model_name:
            raise RuntimeError("Missing MODEL_NAME")
        prompt = (
            "You are an RL training optimizer for Linux SRE command environments. "
            "Return STRICT JSON only with keys: ppo, qlearning. "
            "Each must include a scenario and hyperparameters. "
            "Keep values practical for CPU-only training and stable learning.\n\n"
            f"Default scenario: {default_scenario}\n"
            f"Total timesteps budget: {total_timesteps}\n"
            "Allowed scenarios: log_analysis, permission_repair, process_recovery, "
            "cascading_db_failure, disk_space_crisis, cron_job_failure, nginx_misconfiguration, "
            "security_incident, memory_leak, network_troubleshooting, full_incident"
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Output valid JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        content = str(response.choices[0].message.content or "").strip()
        plan = _extract_json(content)
        plan["source"] = "llm"
    except Exception as exc:
        print(f"[WARN] LLM guidance unavailable, using defaults: {exc}")
        plan = fallback

    return _sanitize_training_plan(plan, default_scenario)


def get_llm_candidate_plans(default_scenario: str, total_timesteps: int, trials: int) -> List[Dict[str, Any]]:
    """Ask the LLM for multiple candidate plans for AutoRL search."""
    trials = max(1, min(trials, 8))
    fallback = [get_llm_training_plan(default_scenario, total_timesteps) for _ in range(trials)]

    try:
        client = _build_openai_client()
        model_name = os.environ.get("MODEL_NAME", "").strip()
        if not model_name:
            raise RuntimeError("Missing MODEL_NAME")
        prompt = (
            "Return STRICT JSON with structure: {\"candidates\": [ ... ]}. "
            "Provide exactly "
            f"{trials} candidates, each candidate containing keys ppo and qlearning with hyperparameters. "
            "Optimize for high final task score on Linux SRE command environments and CPU-friendly training.\n\n"
            f"Timesteps budget: {total_timesteps}\n"
            f"Default scenario: {default_scenario}\n"
            "Allowed scenarios: " + ", ".join(sorted(ALLOWED_SCENARIOS))
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Output valid JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        content = str(response.choices[0].message.content or "").strip()
        payload = _extract_json(content)
        raw_candidates = payload.get("candidates", [])
        candidates: List[Dict[str, Any]] = []
        for raw in raw_candidates:
            if isinstance(raw, dict):
                raw["source"] = "llm-candidate"
                candidates.append(_sanitize_training_plan(raw, default_scenario))

        if not candidates:
            return fallback
        while len(candidates) < trials:
            candidates.append(candidates[-1])
        return candidates[:trials]
    except Exception as exc:
        print(f"[WARN] Candidate-plan generation failed, using fallback plans: {exc}")
        return fallback


def evaluate_qlearning_agent(agent: Any, scenarios: List[str], episodes_per_scenario: int = 1) -> float:
    """Evaluate a Q-learning agent across scenarios and return average final score [0,1]."""
    scores: List[float] = []
    for scenario in scenarios:
        for _ in range(max(1, episodes_per_scenario)):
            env = gym.make("ChaosLab-v0", scenario=scenario)
            obs, _ = env.reset()
            done = False
            final_score = 0.0
            while not done:
                action_arr, _ = agent.predict(obs, deterministic=True)
                action = int(action_arr)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_score = float(info.get("task_score", final_score))
            summary_fn = getattr(env.unwrapped, "get_episode_summary", None)
            if callable(summary_fn):
                summary = summary_fn()
                final_score = float(summary.get("final_score", final_score))
            env.close()
            scores.append(min(max(final_score, 0.0), 1.0))

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def autotune_qlearning_with_llm(total_timesteps: int, scenario: str, trials: int = 3) -> Dict[str, Any]:
    """Run LLM-proposed multi-trial Q-learning search and return best config."""
    from src.qlearning_agent import train_qlearning as _train_ql

    search_trials = max(1, trials)
    search_steps = max(300, total_timesteps // 5)
    search_episodes = max(30, search_steps // 50)
    eval_scenarios = [s for s in DEFAULT_EVAL_SCENARIOS if s in ALLOWED_SCENARIOS]

    candidates = get_llm_candidate_plans(scenario, total_timesteps, search_trials)

    best_score = float("-inf")
    best_cfg: Optional[Dict[str, Any]] = None
    best_scenario = scenario

    print(f"\n[AutoRL] Running {search_trials} Q-learning trials (episodes={search_episodes})")
    print(f"[AutoRL] Evaluation scenarios: {', '.join(eval_scenarios)}")

    for idx, candidate in enumerate(candidates, start=1):
        qcfg = dict(candidate.get("qlearning", {}))
        candidate_scenario = _sanitize_scenario(str(qcfg.pop("scenario", scenario)), scenario)

        agent = _train_ql(
            scenario=candidate_scenario,
            episodes=search_episodes,
            alpha=float(qcfg.get("alpha", 0.15)),
            gamma=float(qcfg.get("gamma", 0.95)),
            epsilon_start=float(qcfg.get("epsilon_start", 1.0)),
            epsilon_end=float(qcfg.get("epsilon_end", 0.05)),
            epsilon_decay=float(qcfg.get("epsilon_decay", 0.99)),
            verbose=False,
        )
        score = evaluate_qlearning_agent(agent, eval_scenarios, episodes_per_scenario=1)
        print(
            f"[AutoRL] Trial {idx}/{search_trials} | "
            f"scenario={candidate_scenario} alpha={qcfg.get('alpha', 0.15)} "
            f"gamma={qcfg.get('gamma', 0.95)} score={score:.3f}"
        )

        if score > best_score:
            best_score = score
            best_cfg = qcfg
            best_scenario = candidate_scenario

    if best_cfg is None:
        best_cfg = {
            "alpha": 0.15,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.99,
        }

    print(f"[AutoRL] Best Q-learning config score={best_score:.3f} on scenario={best_scenario}")
    return {
        "scenario": best_scenario,
        "cfg": best_cfg,
        "score": best_score,
    }


def train_ppo(
    total_timesteps: int,
    scenario: str = "full_incident",
    ppo_overrides: Optional[Dict[str, Any]] = None,
):
    """Train PPO (Neural Network) model."""
    from stable_baselines3 import PPO

    cfg = {
        "learning_rate": 3e-4,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
    }
    if ppo_overrides:
        cfg.update(ppo_overrides)

    print(f"\n{'='*60}")
    print(f"  TRAINING: PPO (Proximal Policy Optimization)")
    print(f"  Type: Neural Network (MLP)")
    print(f"  Steps: {total_timesteps:,}")
    print(f"  Scenario: {scenario}")
    print(f"  Params: lr={cfg['learning_rate']} n_steps={cfg['n_steps']} batch={cfg['batch_size']} epochs={cfg['n_epochs']}")
    print(f"{'='*60}\n")

    env = gym.make("ChaosLab-v0", scenario=scenario)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        ent_coef=cfg["ent_coef"],
        verbose=1,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start_time

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "ppo_model")
    model.save(out_path)

    file_size = os.path.getsize(out_path + ".zip") / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  ✓ PPO TRAINING COMPLETE")
    print(f"  Time:  {elapsed:.1f}s")
    print(f"  Saved: {out_path}.zip ({file_size:.2f} MB)")
    print(f"{'='*60}\n")

    env.close()
    return out_path + ".zip"


def train_qlearning(
    total_timesteps: int,
    scenario: str = "full_incident",
    ql_overrides: Optional[Dict[str, Any]] = None,
):
    """Train Tabular Q-Learning (NO neural network)."""
    from src.qlearning_agent import train_qlearning as _train_ql

    cfg = {
        "alpha": 0.15,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.99,
    }
    if ql_overrides:
        cfg.update(ql_overrides)

    # Convert timesteps to episodes (approx 50 steps per episode)
    episodes = max(50, total_timesteps // 50)

    print(f"\n{'='*60}")
    print(f"  TRAINING: TABULAR Q-LEARNING")
    print(f"  Type: Q-Value Lookup Table (NO neural network)")
    print(f"  Episodes: {episodes}")
    print(f"  Scenario: {scenario}")
    print(f"  Params: alpha={cfg['alpha']} gamma={cfg['gamma']} eps=({cfg['epsilon_start']}->{cfg['epsilon_end']}) decay={cfg['epsilon_decay']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    agent = _train_ql(
        scenario=scenario,
        episodes=episodes,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay=cfg["epsilon_decay"],
        verbose=True,
    )

    elapsed = time.time() - start_time

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "qlearning_model.json")
    agent.save(out_path)

    file_size = os.path.getsize(out_path) / 1024  # KB for Q-tables

    stats = agent.stats()

    print(f"\n{'='*60}")
    print(f"  ✓ Q-LEARNING TRAINING COMPLETE")
    print(f"  Time:  {elapsed:.1f}s")
    print(f"  Saved: {out_path} ({file_size:.1f} KB)")
    print(f"  Q-table entries: {stats['table_entries']:,}")
    print(f"  Unique states:   {stats['unique_states']:,}")
    print(f"{'='*60}\n")

    return out_path


ALGO_TRAINERS = {
    "ppo": train_ppo,
    "qlearning": train_qlearning,
}


def train_all(
    total_timesteps: int,
    scenario: str = "full_incident",
    llm_guided: bool = False,
    llm_autorl_trials: int = DEFAULT_AUTORL_TRIALS,
):
    """Train all available algorithms sequentially."""
    print("\n" + "=" * 60)
    print("   CHAOSLAB MULTI-MODEL TRAINING PIPELINE")
    print(f"   Models: PPO (Neural Net) + Q-Learning (Tabular)")
    print(f"   Steps: {total_timesteps:,}")
    print(f"   Guidance: {'LLM-Tuned' if llm_guided else 'Default'}")
    if llm_guided:
        print(f"   AutoRL Trials (Q-Learning): {llm_autorl_trials}")
    print("=" * 60)

    results = {}
    total_start = time.time()

    plan = None
    ql_autorl_result: Optional[Dict[str, Any]] = None
    if llm_guided:
        plan = get_llm_training_plan(default_scenario=scenario, total_timesteps=total_timesteps)
        print(f"[INFO] Training plan source: {plan.get('source', 'unknown')}")
        if llm_autorl_trials > 1:
            ql_autorl_result = autotune_qlearning_with_llm(
                total_timesteps=total_timesteps,
                scenario=scenario,
                trials=llm_autorl_trials,
            )

    for algo, trainer in ALGO_TRAINERS.items():
        try:
            if algo == "ppo" and plan:
                ppo_cfg = dict(plan.get("ppo", {}))
                ppo_scenario = str(ppo_cfg.pop("scenario", scenario))
                path = trainer(total_timesteps, ppo_scenario, ppo_cfg)
            elif algo == "qlearning" and ql_autorl_result:
                ql_cfg = dict(ql_autorl_result.get("cfg", {}))
                ql_scenario = str(ql_autorl_result.get("scenario", scenario))
                path = trainer(total_timesteps, ql_scenario, ql_cfg)
            elif algo == "qlearning" and plan:
                ql_cfg = dict(plan.get("qlearning", {}))
                ql_scenario = str(ql_cfg.pop("scenario", scenario))
                path = trainer(total_timesteps, ql_scenario, ql_cfg)
            else:
                path = trainer(total_timesteps, scenario)
            results[algo] = {"status": "success", "path": path}
        except Exception as e:
            print(f"[ERROR] Failed to train {algo}: {e}")
            results[algo] = {"status": "failed", "error": str(e)}

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("   TRAINING SUMMARY")
    print("=" * 60)
    for algo, info in results.items():
        status = "✓" if info["status"] == "success" else "✗"
        detail = info.get("path", info.get("error", ""))
        print(f"   {status} {algo.upper():>12s}  →  {detail}")
    print(f"\n   Total Time: {total_elapsed:.1f}s")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChaosLab AI Training Pipeline")
    parser.add_argument("--algo", type=str, default=None,
                        choices=list(ALGO_TRAINERS.keys()),
                        help="Train a specific algorithm (default: train all)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Training timesteps (default: {DEFAULT_STEPS:,})")
    parser.add_argument("--scenario", type=str, default="full_incident",
                        help="Scenario to train on (default: full_incident)")
    parser.add_argument("--llm-guided", action="store_true",
                        help="Use OpenAI-guided hyperparameter tuning via API_BASE_URL/MODEL_NAME/HF_TOKEN")
    parser.add_argument("--llm-autorl-trials", type=int, default=DEFAULT_AUTORL_TRIALS,
                        help=f"AutoRL search trials for Q-learning when --llm-guided (default: {DEFAULT_AUTORL_TRIALS})")

    args = parser.parse_args()

    if args.algo:
        if args.llm_guided:
            plan = get_llm_training_plan(default_scenario=args.scenario, total_timesteps=args.steps)
            if args.algo == "ppo":
                ppo_cfg = dict(plan.get("ppo", {}))
                ppo_scenario = str(ppo_cfg.pop("scenario", args.scenario))
                train_ppo(args.steps, ppo_scenario, ppo_cfg)
            else:
                if args.llm_autorl_trials > 1:
                    tuned = autotune_qlearning_with_llm(
                        total_timesteps=args.steps,
                        scenario=args.scenario,
                        trials=args.llm_autorl_trials,
                    )
                    train_qlearning(
                        args.steps,
                        str(tuned.get("scenario", args.scenario)),
                        dict(tuned.get("cfg", {})),
                    )
                else:
                    ql_cfg = dict(plan.get("qlearning", {}))
                    ql_scenario = str(ql_cfg.pop("scenario", args.scenario))
                    train_qlearning(args.steps, ql_scenario, ql_cfg)
        else:
            ALGO_TRAINERS[args.algo](args.steps, args.scenario)
    else:
        train_all(
            args.steps,
            args.scenario,
            llm_guided=args.llm_guided,
            llm_autorl_trials=args.llm_autorl_trials,
        )
