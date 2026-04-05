import asyncio
import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .environment import TrainingEnv
from .scenarios import list_scenarios, detail_scenario, SCENARIO_CATALOG
from .model_registry import get_registry

app = FastAPI(
    title="Linux SRE Environment API",
    description=(
        "OpenEnv-compliant API for the Linux SRE training environment. "
        "Supports legacy difficulty-based tasks and composable scenarios "
        "with cascading fault injection."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

backends: Dict[str, TrainingEnv] = {}
counter = 0


# ======================================================================
#  REQUEST / RESPONSE MODELS
# ======================================================================

class ResetPayload(BaseModel):
    scenario: str = Field(
        default="log_analysis", description="Scenario key to load")
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility")


class StepPayload(BaseModel):
    action: str = Field(description="Shell command to execute")


class ResetOut(BaseModel):
    env_id: str
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepOut(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ======================================================================
#  HEALTH + TASKS
# ======================================================================

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "linux-sre-env", "version": "2.0.0"}


@app.get("/api/v1/tasks")
async def list_tasks():
    return {
        "tasks": TrainingEnv.avail_tasks(),
        "details": {
            key: TrainingEnv.task_details(key)
            for key in TrainingEnv.avail_tasks()
        }
    }


@app.get("/api/v1/tasks/{key}")
async def get_task(key: str):
    info = TrainingEnv.task_details(key)
    if not info:
        raise HTTPException(status_code=404, detail=f"Task '{key}' not found")
    return info


# ======================================================================
#  SCENARIOS
# ======================================================================

@app.get("/api/v1/scenarios")
async def get_scenarios():
    """List all available scenarios with metadata."""
    return {"scenarios": list_scenarios()}


@app.get("/api/v1/scenarios/{key}")
async def get_scenario(key: str):
    """Get detailed info for a single scenario."""
    if key not in SCENARIO_CATALOG:
        raise HTTPException(
            status_code=404, detail=f"Scenario '{key}' not found")
    return detail_scenario(key)


# ======================================================================
#  AI MODELS
# ======================================================================

@app.get("/api/v1/models")
async def list_models():
    """List all available AI models with metadata."""
    registry = get_registry()
    return {"models": registry.list_models()}


@app.get("/api/v1/models/{name}")
async def get_model(name: str):
    """Get info for a specific AI model."""
    registry = get_registry()
    info = registry.get_model_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return info


# ======================================================================
#  ENVIRONMENT LIFECYCLE
# ======================================================================

@app.post("/api/v1/env/reset")
async def reset(req: ResetPayload):
    global counter
    try:
        env = TrainingEnv(scenario=req.scenario)
        eid = f"env_{counter}"
        counter += 1
        backends[eid] = env
        res = env.reset()
        return ResetOut(env_id=eid, observation=res["observation"], info=res["info"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/env/{env_id}/step")
async def step(env_id: str, req: StepPayload):
    if env_id not in backends:
        raise HTTPException(
            status_code=404, detail=f"Environment '{env_id}' not found")
    env = backends[env_id]
    res = env.step(req.action)
    return StepOut(
        observation=res["observation"],
        reward=res["reward"],
        done=res["done"],
        info=res["info"],
    )


@app.get("/api/v1/env/{env_id}/state")
async def get_state(env_id: str):
    if env_id not in backends:
        raise HTTPException(
            status_code=404, detail=f"Environment '{env_id}' not found")
    return backends[env_id].dump()


@app.delete("/api/v1/env/{env_id}")
async def delete_env(env_id: str):
    if env_id not in backends:
        raise HTTPException(
            status_code=404, detail=f"Environment '{env_id}' not found")
    del backends[env_id]
    return {"status": "deleted", "env_id": env_id}


@app.get("/api/v1/env")
async def list_envs():
    return {
        "count": len(backends),
        "environments": {
            eid: {
                "task": env.task.nm,
                "difficulty": env.difficulty,
                "score": env.score,
                "step": env.step_count,
                "done": env.finished,
            }
            for eid, env in backends.items()
        },
    }


# ======================================================================
#  WEBSOCKET — live terminal streaming
# ======================================================================

class ConnectionManager:
    """Manages WebSocket connections for live terminal streaming."""

    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, env_id: str, ws: WebSocket):
        await ws.accept()
        if env_id not in self.active:
            self.active[env_id] = []
        self.active[env_id].append(ws)

    def disconnect(self, env_id: str, ws: WebSocket):
        if env_id in self.active:
            self.active[env_id] = [w for w in self.active[env_id] if w != ws]

    async def broadcast(self, env_id: str, data: dict):
        if env_id in self.active:
            for ws in self.active[env_id]:
                try:
                    await ws.send_json(data)
                except Exception:
                    pass


ws_manager = ConnectionManager()


@app.websocket("/ws/env/{env_id}")
async def ws_terminal(ws: WebSocket, env_id: str):
    """
    WebSocket endpoint for live terminal interaction.

    Client sends: {"action": "step", "command": "ps aux"}
    Server sends: {"type": "output", "command": "...", "output": "...", "score": 0.5, ...}

    Also supports: {"action": "reset"}, {"action": "state"}
    """
    if env_id not in backends:
        await ws.close(code=4004, reason=f"Environment '{env_id}' not found")
        return

    await ws_manager.connect(env_id, ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            action = msg.get("action", "")
            env = backends.get(env_id)
            if not env:
                await ws.send_json({"type": "error", "message": "Environment not found"})
                break

            if action == "step":
                command = msg.get("command", "")
                if not command:
                    await ws.send_json({"type": "error", "message": "Missing 'command'"})
                    continue
                res = env.step(command)
                payload = {
                    "type": "output",
                    "command": command,
                    "output": res["info"].get("command_output", ""),
                    "score": res["info"]["task_score"],
                    "reward": res["reward"],
                    "step": res["info"]["step"],
                    "max_steps": res["info"]["max_steps"],
                    "done": res["done"],
                    "exit_code": res["info"]["exit_code"],
                }
                await ws_manager.broadcast(env_id, payload)

            elif action == "reset":
                res = env.reset()
                await ws.send_json({
                    "type": "reset",
                    "observation": res["observation"],
                    "info": res["info"],
                })

            elif action == "state":
                await ws.send_json({
                    "type": "state",
                    "data": env.dump(),
                })

            else:
                await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        ws_manager.disconnect(env_id, ws)


class AgentRunPayload(BaseModel):
    agent_type: str = Field(description="Agent type: llm or rl")
    model_name: str = Field(default="ppo", description="Model name for RL agent (ppo, a2c, heuristic)")


async def run_agent_in_background(env_id: str, agent_type: str, model_name: str = "ppo"):
    """Background task to run an autonomous agent and stream to websockets."""
    env = backends.get(env_id)
    if not env:
        return

    try:
        if agent_type == "llm":
            from .agent import LLMAgent
            import os
            
            # Send a system notice that LLM is booting
            await ws_manager.broadcast(env_id, {"type": "system", "text": "Booting Autonomous LLM via litellm..."})
            await asyncio.sleep(1)
            
            agent = LLMAgent(model="openai/gpt-4o", api_key=os.environ.get("OPENAI_API_KEY", ""), verbose=False)
            
            # Since LLMAgent normally runs a while loop on its own and blocks, 
            # we need to simulate its turns or hook into it.
            # To intercept and stream to WS, we inject a custom hook into agent.step or just run it via run_in_executor
            
            # Actually, `agent.solve` takes scenario string and uses a new env.
            # We want it to use our existing env.
            # We can run the loop manually:
            obs_dict = env.dump()
            obs_str = f"Dir: {obs_dict.get('cwd', '/')}\nFilesystem:\n{obs_dict.get('filesystem','')}\nProcesses:\n{obs_dict.get('processes','')}"
            
            done = False
            step = 0
            while not done and step < env.limit:
                # LLM takes a few seconds to think
                await ws_manager.broadcast(env_id, {"type": "system", "text": "LLM is thinking..."})
                await asyncio.sleep(2.0) 
                
                # We could call real LLM here, but since user permitted dummy for RL maybe dummy for LLM without API key is fine
                # Or we can just try to run real litellm.
                # If API key is empty, LLMAgent raises exception.
                try:
                    command = agent.turn(obs_str)
                except Exception as e:
                    await ws_manager.broadcast(env_id, {"type": "error", "message": f"LLM Error: {str(e)}"})
                    break
                    
                # Broadcast the input exactly like human
                await ws_manager.broadcast(env_id, {"type": "input", "text": command})
                
                res = env.step(command)
                payload = {
                    "type": "output",
                    "command": command,
                    "output": res["info"].get("command_output", ""),
                    "score": res["info"]["task_score"],
                    "reward": res["reward"],
                    "step": res["info"]["step"],
                    "max_steps": res["info"]["max_steps"],
                    "done": res["done"],
                    "exit_code": res["info"]["exit_code"],
                }
                await ws_manager.broadcast(env_id, payload)
                obs_str = res["info"].get("command_output", "")
                done = res["done"]
                step += 1

        elif agent_type == "rl":
            from src.rl_env import ChaosLabEnv

            registry = get_registry()
            display_name = registry.get_display_name(model_name)

            await ws_manager.broadcast(env_id, {"type": "system", "text": f"Booting AI Agent: [{display_name}]..."})
            await asyncio.sleep(1)

            # Hijack the actively running sandbox environment
            rl_env = ChaosLabEnv()
            rl_env.hijack_env(env)
            catalog = rl_env.get_action_catalog()

            # Load the selected model from registry
            model, model_info = registry.load_model(model_name)
            actual_name = model_info.display_name
            await ws_manager.broadcast(env_id, {
                "type": "system",
                "text": f"[Model Loaded: {actual_name}] Algorithm: {model_info.algorithm}"
            })

            # Reset heuristic state if applicable
            if hasattr(model, 'reset'):
                model.reset()

            done = False
            while not done and env.step_count < env.limit:
                await asyncio.sleep(1.2)  # Simulate thinking delay

                # Get observation and predict
                obs = rl_env.get_current_obs()

                # Stream agent reasoning for heuristic
                if hasattr(model, 'get_reasoning'):
                    reasoning = model.get_reasoning(obs)
                    await ws_manager.broadcast(env_id, {
                        "type": "system",
                        "text": f"[{actual_name}] {reasoning}"
                    })
                    await asyncio.sleep(0.5)

                action_arr, _ = model.predict(obs, deterministic=True)
                action = int(action_arr)
                command = catalog[action]

                # Broadcast the input
                await ws_manager.broadcast(env_id, {"type": "input", "text": command})

                # Step the actual TrainingEnv
                res = env.step(command)
                payload = {
                    "type": "output",
                    "command": command,
                    "output": res["info"].get("command_output", ""),
                    "score": res["info"]["task_score"],
                    "reward": res["reward"],
                    "step": res["info"]["step"],
                    "max_steps": res["info"]["max_steps"],
                    "done": res["done"],
                    "exit_code": res["info"]["exit_code"],
                    "model_name": model_name,
                }
                await ws_manager.broadcast(env_id, payload)
                done = res["done"]

    except Exception as e:
        await ws_manager.broadcast(env_id, {"type": "error", "message": f"Agent crashed: {str(e)}"})


@app.post("/api/v1/env/{env_id}/agent_run")
async def start_agent_run(env_id: str, req: AgentRunPayload, background_tasks: BackgroundTasks):
    if env_id not in backends:
        raise HTTPException(status_code=404, detail="Environment not found")

    background_tasks.add_task(run_agent_in_background, env_id, req.agent_type, req.model_name)
    return {"status": "started", "agent_type": req.agent_type, "model_name": req.model_name}


# ======================================================================
#  AGENT ARENA — compare models on same scenario
# ======================================================================

class ArenaPayload(BaseModel):
    scenario: str = Field(description="Scenario key to run")
    commands_a: List[str] = Field(description="Commands for Agent A")
    commands_b: List[str] = Field(description="Commands for Agent B")
    label_a: str = Field(default="Agent A")
    label_b: str = Field(default="Agent B")
    type_a: str = Field(default="script", description="Type of agent: script, llm, rl")
    type_b: str = Field(default="script", description="Type of agent: script, llm, rl")
    model_a: str = Field(default="ppo", description="Model name for Agent A (if type is rl)")
    model_b: str = Field(default="heuristic", description="Model name for Agent B (if type is rl)")


@app.post("/api/v1/arena/run")
async def arena_run(req: ArenaPayload):
    """Run two command sequences or autonomous agents on the same scenario and compare scores."""
    results = {}
    
    agent_configs = [
        (req.label_a, req.type_a, req.commands_a, req.model_a),
        (req.label_b, req.type_b, req.commands_b, req.model_b),
    ]

    for label, agent_type, commands, model_name_for_agent in agent_configs:
        history = []
        final_score = 0.0
        steps_used = 0
        completed = False

        if agent_type == "llm":
            # Run Autonomous LLM (fallback to error if no API key/fails)
            try:
                from .agent import LLMAgent
                import os
                agent = LLMAgent(model="openai/gpt-4o", api_key=os.environ.get("OPENAI_API_KEY", ""), verbose=False)
                res = agent.solve(scenario=req.scenario)
                for i, turn in enumerate(res.get("turns", [])):
                    history.append({
                        "command": turn["command"],
                        "score": turn["score"],
                        "exit_code": 0, # Simplify
                    })
                final_score = res["final_score"]
                steps_used = res["steps_used"]
                completed = res["completed"]
            except Exception as e:
                history.append({
                    "command": f"LLM Initialization Failed: {str(e)}",
                    "score": 0.0,
                    "exit_code": 1
                })
        
        elif agent_type == "rl":
            # Run RL Agent using the model registry
            try:
                import gymnasium as gym
                import src.rl_env

                # Determine which model to load for this agent slot
                agent_model_name = model_name_for_agent

                registry = get_registry()
                model, model_info = registry.load_model(agent_model_name)

                env = gym.make("ChaosLab-v0", scenario=req.scenario)
                obs, _ = env.reset()

                # Reset heuristic state if applicable
                if hasattr(model, 'reset'):
                    model.reset()

                done = False
                while not done:
                    action_arr, _ = model.predict(obs, deterministic=True)
                    action = int(action_arr)

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    history.append({
                        "command": info.get("command", f"Action {action}"),
                        "score": info.get("task_score", 0.0),
                        "exit_code": info.get("exit_code", 0),
                        "model": agent_model_name,
                    })

                summary = env.unwrapped.get_episode_summary()
                final_score = summary["final_score"]
                steps_used = summary["steps_used"]
                completed = summary["completed"]
                env.close()
            except Exception as e:
                history.append({
                    "command": f"RL Initialization Failed ({agent_model_name}): {str(e)}",
                    "score": 0.0,
                    "exit_code": 1
                })

        else:
            # Standard Script looping
            env = TrainingEnv(scenario=req.scenario)
            env.reset()
            for cmd in commands:
                res = env.step(cmd)
                history.append({
                    "command": cmd,
                    "score": res["info"]["task_score"],
                    "exit_code": res["info"]["exit_code"],
                })
                if res["done"]:
                    break
            final_score = env.score
            steps_used = env.step_count
            completed = env.finished

        results[label] = {
            "final_score": final_score,
            "steps_used": steps_used,
            "completed": completed,
            "history": history,
        }

    # determine winner
    scores = {k: v["final_score"] for k, v in results.items()}
    winner = max(scores, key=scores.get)
    if len(set(scores.values())) == 1:
        winner = "tie"

    return {
        "scenario": req.scenario,
        "results": results,
        "winner": winner,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
