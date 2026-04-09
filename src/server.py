import json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .environment import TrainingEnv
from .scenarios import list_scenarios, detail_scenario, SCENARIO_CATALOG

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

@app.get("/")
async def root():
    return {"status": "ok", "service": "linux-sre-env", "version": "2.0.0", "docs": "/docs"}


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


# ======================================================================
#  TOP-LEVEL OpenEnv ALIASES  (validators may hit /reset, /step, /state)
# ======================================================================

@app.post("/reset")
async def reset_alias(req: ResetPayload):
    return await reset(req)


@app.post("/step/{env_id}")
async def step_alias(env_id: str, req: StepPayload):
    return await step(env_id, req)


@app.get("/state/{env_id}")
async def state_alias(env_id: str):
    return await get_state(env_id)


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
