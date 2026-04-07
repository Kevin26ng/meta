"""
Demonstration script showing how LLMs and reinforcement Learning algorithms
can co-exist in the ChaosLab architecture.

Usage:
    python demo_agents.py --mode rl
    python demo_agents.py --mode llm

Requirements:
    pip install -r requirements.txt (installs gymnasium and stable-baselines3)
    Set API_BASE_URL, MODEL_NAME, HF_TOKEN for LLM mode
"""

import argparse
import os
from typing import List


def run_llm_agent(scenario: str, model: str):
    """Run an LLM agent on the specified scenario."""
    from src.agent import LLMAgent
    
    print("\n" + "="*60)
    print(f"[STARTING LLM AGENT] (Model: {model})")
    print("="*60)
    
    api_key = os.environ.get("HF_TOKEN", "")
    base_url = os.environ.get("API_BASE_URL", "")
    effective_model = model or os.environ.get("MODEL_NAME", "")
    missing: List[str] = []
    if not base_url:
        missing.append("API_BASE_URL")
    if not effective_model:
        missing.append("MODEL_NAME")
    if not api_key:
        missing.append("HF_TOKEN")
    if missing:
        print(f"Warning: Missing required environment variables: {', '.join(missing)}")
        print("LLM mode requires API_BASE_URL, MODEL_NAME, and HF_TOKEN.")
        return
        
    try:
        agent = LLMAgent(model=effective_model, api_key=api_key, base_url=base_url, verbose=True)
        result = agent.solve(scenario=scenario)
        
        print("\n[LLM EPISODE FINISHED]")
        print(f"Final Score: {result['final_score']}")
        print(f"Steps Used:  {result['steps_used']} / {result['max_steps']}")
        print(f"Completed:   {result['completed']}")
    except ImportError:
        print("Error: Missing dependencies. Run: pip install -r requirements.txt")


def run_rl_agent(scenario: str, algo: str = "PPO", timesteps: int = 5000):
    """Train or run an RL agent on the specified scenario."""
    print("\n" + "="*60)
    print(f"[STARTING REINFORCEMENT LEARNING AGENT] (Algo: {algo})")
    print("="*60)
    
    import gymnasium as gym
    import src.rl_env  # This registers ChaosLab-v0
    from src.model_registry import get_registry
    
    # Create the environment
    env = gym.make("ChaosLab-v0", scenario=scenario, render_mode="human")
    
    model_key = algo.lower()
    
    if model_key == "qlearning":
        # Q-Learning: Tabular, NO neural network
        registry = get_registry()
        model, model_info = registry.load_model("qlearning")
        print(f"\n[1] Loaded: {model_info.display_name} ({model_info.algorithm})")
        if hasattr(model, 'reset'):
            model.reset()
    elif model_key == "ppo":
        # PPO: Neural Network
        registry = get_registry()
        model, model_info = registry.load_model("ppo")
        print(f"\n[1] Loaded: {model_info.display_name} ({model_info.algorithm})")
    elif model_key == "heuristic":
        # Heuristic: Hand-coded expert
        registry = get_registry()
        model, model_info = registry.load_model("heuristic")
        print(f"\n[1] Loaded: {model_info.display_name} ({model_info.algorithm})")
        if hasattr(model, 'reset'):
            model.reset()
    else:
        print(f"Unknown algo: {algo}. Falling back to random.")
        model = None
    
    print(f"\n[2] Running agent on scenario: {scenario}...")
    obs, info = env.reset()
    done = False
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
            
    # 3. Summary Output
    if hasattr(env.unwrapped, "get_episode_summary"):
        summary = env.unwrapped.get_episode_summary()
        print("\n[RL EPISODE FINISHED]")
        print(f"Final Score:    {summary['final_score']}")
        print(f"Steps Used:     {summary['steps_used']} / {summary['max_steps']}")
        print(f"Cumulative Rwd: {summary['episode_reward']:.2f}")
        print(f"History Length: {len(summary['commands_executed'])}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChaosLab AI Agents Demo")
    parser.add_argument("--mode", choices=["llm", "rl"], required=True,
                        help="Choose which type of agent to run (llm or rl)")
    parser.add_argument("--scenario", default="disk_space_crisis",
                        help="Which scenario to run (e.g. disk_space_crisis, full_incident)")
    parser.add_argument("--llm_model", default="",
                        help="LLM model to use (for LLM mode). Defaults to MODEL_NAME env var.")
    parser.add_argument("--rl_algo", type=str, default="PPO",
                        choices=["PPO", "QLEARNING", "HEURISTIC"],
                        help="RL algorithm to use (PPO=neural net, QLEARNING=tabular, HEURISTIC=rule-based)")
    parser.add_argument("--timesteps", type=int, default=5000,
                        help="How many timesteps to train (for RL mode)")
                        
    args = parser.parse_args()
    
    if args.mode == "llm":
        run_llm_agent(scenario=args.scenario, model=args.llm_model)
    elif args.mode == "rl":
        run_rl_agent(scenario=args.scenario, algo=args.rl_algo, timesteps=args.timesteps)
