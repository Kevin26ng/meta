import asyncio
import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .environment import TrainingEnv
from .scenarios import list_scenarios, SCENARIO_CATALOG

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
    difficulty: str = Field(
        default="medium", description="Task difficulty or scenario key")
    scenario: Optional[str] = Field(
        default=None, description="Scenario key (overrides difficulty)")
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
    return list_scenarios()[key]


# ======================================================================
#  ENVIRONMENT LIFECYCLE
# ======================================================================

@app.post("/api/v1/env/reset")
async def reset(req: ResetPayload):
    global counter
    try:
        env = TrainingEnv(difficulty=req.difficulty, scenario=req.scenario)
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


# ======================================================================
#  AGENT ARENA — compare models on same scenario
# ======================================================================

class ArenaPayload(BaseModel):
    scenario: str = Field(description="Scenario key to run")
    commands_a: List[str] = Field(description="Commands for Agent A")
    commands_b: List[str] = Field(description="Commands for Agent B")
    label_a: str = Field(default="Agent A")
    label_b: str = Field(default="Agent B")


@app.post("/api/v1/arena/run")
async def arena_run(req: ArenaPayload):
    """Run two command sequences on the same scenario and compare scores."""
    results = {}
    for label, commands in [(req.label_a, req.commands_a), (req.label_b, req.commands_b)]:
        env = TrainingEnv(scenario=req.scenario)
        env.reset()
        history = []
        for cmd in commands:
            res = env.step(cmd)
            history.append({
                "command": cmd,
                "score": res["info"]["task_score"],
                "exit_code": res["info"]["exit_code"],
            })
            if res["done"]:
                break
        results[label] = {
            "final_score": env.score,
            "steps_used": env.step_count,
            "completed": env.finished,
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
