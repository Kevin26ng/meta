# ChaosLab: AI SRE Training Environment

ChaosLab is a production-style training environment for AI agents to practice Linux/SRE incident response.
It combines a virtual Linux system, objective scoring, RL agents, and optional OpenAI-powered assistance.

## Why this project is strong for hackathons

- Real agentic workflow: agents execute shell commands, inspect outputs, and recover systems.
- Objective scoring: scenarios return measurable progress and completion scores.
- Practical scope: incident-response workflows map to real SRE tasks.
- Interactive demoability: complete web UI (Hub, Playground, Arena, Builder).

## Core features

- Virtual Linux environment with deterministic reset/step loop.
- 11 scenarios from simple log triage to full cascading incidents.
- Terminal emulator with realistic command flows.
- FastAPI backend with REST and WebSocket streaming.
- Next.js frontend for scenario browsing, live command execution, and benchmarking.
- Multiple agent modes (heuristic, RL, optional LLM guidance).
- Arena mode for side-by-side agent comparisons.
- Root-level inference.py for submission/runtime compatibility.

## Architecture

- Backend: src/server.py, src/environment.py, src/scenarios.py, src/terminal_emulator.py
- Agents: src/agent.py, src/train_ai.py, model registry under models/
- Frontend: frontend/src/app/*
- Inference entrypoint: inference.py

## Quick start

### 1) Prerequisites

- Python 3.11+
- Node.js 18+
- npm

### 2) Install dependencies

```bash
# from repo root
pip install -r requirements.txt

# frontend deps
cd frontend
npm install
cd ..
```

### 3) Start backend

```bash
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```

### 4) Start frontend

```bash
cd frontend
npm run dev
```

### 5) Open the app

- Frontend: http://localhost:3000
- Backend health: http://127.0.0.1:8000/health

## Optional LLM setup

Set these environment variables before starting backend if you want LLM features:

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
HF_TOKEN=<your_api_key>
```

Notes:

- The project uses OpenAI-compatible client flow for LLM calls.
- If unset, non-LLM features still work.

## How to use the website

1. Hub (/): pick a scenario.
2. Playground (/playground): initialize sandbox and run commands.
3. Arena (/arena): compare agent performance on the same scenario.
4. Builder (/builder): inspect and compose scenario configurations.

Typical Playground loop:

1. Initialize/reset scenario sandbox.
2. Run diagnostic commands (ls, ps, cat, grep, etc.).
3. Apply fixes (chmod, service restart, config corrections, cleanup).
4. Track objective/score progress until completion.

## API quick checks

```bash
# health
curl http://127.0.0.1:8000/health

# list scenarios
curl http://127.0.0.1:8000/api/v1/scenarios

# reset env
curl -X POST http://127.0.0.1:8000/api/v1/env/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario": "log_analysis"}'
```

## Training and inference

```bash
# training options
python src/train_ai.py --help

# submission/runtime entrypoint
python inference.py --help
```

## Submission/readiness notes

- Root inference.py is present.
- Backend and frontend run locally with live interaction.
- Scenario APIs and WebSocket stream are active.
- Arena and model registry endpoints are integrated in UI.

## Key files

```text
src/
  environment.py
  scenarios.py
  terminal_emulator.py
  agent.py
  server.py
  train_ai.py
frontend/
  src/app/page.tsx
  src/app/playground/page.tsx
  src/app/arena/page.tsx
  src/app/builder/page.tsx
inference.py
openenv.yaml
```

## Troubleshooting

- Port 8000 already in use: stop old backend process and restart.
- Frontend says backend unavailable: verify http://127.0.0.1:8000/health.
- LLM unavailable in UI: verify API_BASE_URL, MODEL_NAME, HF_TOKEN.

## License

Provided for hackathon development and evaluation.
