# ChaosLab — AI SRE Training Environment

## What Is This?

ChaosLab is a simulated Linux server that AI agents can connect to and practice fixing real infrastructure problems — like restarting a crashed database, cleaning up a full disk, or tracing a cascading failure across multiple services.

Think of it as a **flight simulator for AI system administrators**. No real servers are touched. Everything runs in pure Python.

---

## The Problem

Companies want AI that can manage servers, but:

- You can't let untrained AI touch real production systems
- There's no standardized way to test if an AI can actually fix a server
- Most AI benchmarks test chatting ability, not hands-on troubleshooting
- Real server issues involve **chain reactions** (one failure causes another), and no existing tool simulates that well

---

## What We Built (Step by Step)

### Step 1: Virtual Linux Filesystem

**File:** `src/virtual_filesystem.py` (~660 lines)

We built a fake Linux server entirely in Python dictionaries. No actual OS calls.

What it simulates:

- **30+ files** — app logs, nginx logs, syslog, auth logs, cron logs, config files, scripts, /proc files, SSH keys
- **6 services** — nginx, postgres, redis, app, cron, sshd (each with PID, status, CPU, memory, port)
- **Disk partitions** — /, /boot, /var/log, /tmp with realistic usage numbers
- **Network ports** — 6 ports with LISTEN/CLOSED states
- **Memory** — 8GB RAM with used/free/cached/swap tracking
- **Cron jobs** — 3 scheduled tasks with pass/fail history
- **Users & groups** — root, www-data, postgres, redis, user, nobody
- **Environment variables** — PATH, DATABASE_URL, REDIS_URL, APP_ENV, etc.
- **Firewall rules** — iptables INPUT/FORWARD/OUTPUT chains
- **DNS records** — localhost, db-primary, db-replica, cache-01, api.example.com

Everything resets instantly. No side effects on your actual machine.

---

### Step 2: Terminal Emulator

**File:** `src/terminal_emulator.py` (~1100 lines)

A shell command parser that reads commands and returns realistic Linux output.

**37 supported commands:**

| Category           | Commands                                                                |
| ------------------ | ----------------------------------------------------------------------- |
| File reading       | `cat`, `head`, `tail`, `grep`, `find`, `wc`                             |
| File management    | `ls`, `touch`, `mkdir`, `cp`, `mv`, `rm`, `chmod`                       |
| Process management | `ps`, `kill`, `systemctl`                                               |
| System info        | `top`, `free`, `df`, `du`, `uptime`, `whoami`, `id`, `hostname`, `date` |
| Networking         | `netstat`/`ss`, `curl`, `dig`/`nslookup`, `iptables`                    |
| Logs & scheduling  | `journalctl`, `crontab`                                                 |
| Text processing    | `sort`, `uniq`, `cut`, `tr`, `tee`, `echo`                              |
| Utilities          | `cd`, `pwd`, `env`, `export`, `which`, `mount`, `history`, `clear`      |

**Also supports:**

- Pipes: `cat /var/log/app.log | grep ERROR | wc -l`
- Redirects: `echo "hello" > /tmp/test.txt`
- Append: `echo "line" >> /tmp/test.txt`
- Chaining: `ps; df -h; free -m`
- Variable expansion: `echo $DATABASE_URL`
- Path resolution: relative paths, `~`, `..`
- Quoting: single and double quotes

---

### Step 3: Composable Scenario Engine (Key Differentiator)

**File:** `src/scenarios.py` (~850 lines)

Instead of hardcoded tasks, we built a modular system where failure scenarios are composed from building blocks.

**Fault Injection Primitives (9 types):**

| Fault                     | What It Does                             | Example                       |
| ------------------------- | ---------------------------------------- | ----------------------------- |
| `crash_service`           | Kills a service, closes its port         | postgres crashes              |
| `fill_disk`               | Increases used space on a mount          | /var/log fills to 95%         |
| `corrupt_config`          | Replaces config file content             | nginx points to wrong port    |
| `bad_permissions`         | Sets wrong file permissions              | script becomes non-executable |
| `add_log_flood`           | Appends thousands of log lines           | 5000 error lines in nginx log |
| `kill_port`               | Closes a network port                    | port 8080 becomes CLOSED      |
| `fail_cron`               | Marks a cron job as failed               | backup job shows FAILED       |
| `memory_pressure`         | Raises memory usage                      | 95% RAM used                  |
| `add_unauthorized_access` | Adds brute-force SSH entries to auth log | 50 failed login attempts      |

**Cascading Rules:**

Faults can trigger other faults automatically:

```
Database crashes
  └─→ App can't connect, throws errors
       └─→ Error logs flood /var/log
            └─→ Disk fills up to 95%
                 └─→ App can't write logs, crashes too
```

This is defined declaratively:

```python
CascadeRule(
    condition_fn="service_is_dead",        # IF postgres is dead
    condition_params={"service": "postgres"},
    effect=Fault("disk_fill", ...,         # THEN fill the /var/log disk
                 "fill_disk", {"mount": "/var/log", "fill_mb": 8500}),
)
```

Cascades are evaluated after every command the AI runs, so the environment evolves dynamically.

---

### Step 4: 11 Pre-Built Scenarios

| #   | Scenario                    | Difficulty | Objectives | What Happens                                                                       |
| --- | --------------------------- | ---------- | ---------- | ---------------------------------------------------------------------------------- |
| 1   | **Log Analysis**            | Easy       | 1          | Find timestamp of first 500 error in app logs                                      |
| 2   | **Permission Repair**       | Medium     | 1          | Fix a script that can't execute (chmod)                                            |
| 3   | **Disk Space Crisis**       | Medium     | 2          | /var/log partition is 90%+ full, clean it up                                       |
| 4   | **Cron Job Failure**        | Medium     | 3          | Backup cron failing — create and fix the script                                    |
| 5   | **Process Recovery**        | Hard       | 3          | PostgreSQL crashed, restart and verify                                             |
| 6   | **Nginx Misconfiguration**  | Hard       | 3          | Wrong upstream port after deploy, fix config                                       |
| 7   | **Security Incident**       | Hard       | 3          | Brute-force SSH attack, investigate auth logs                                      |
| 8   | **Memory Leak**             | Hard       | 4          | Server low on RAM, find and restart leaky service                                  |
| 9   | **Network Troubleshooting** | Hard       | 4          | App unreachable — diagnose DNS/ports/firewall                                      |
| 10  | **Cascading DB Failure**    | Expert     | 5          | DB crash → 502s → log flood → disk full → app crash                                |
| 11  | **Full Incident Response**  | Expert     | 6          | Everything broken at once: DB down, disk full, brute-force attack, memory pressure |

Each scenario has:

- Multiple graded objectives (partial credit)
- Hints for the AI
- Cascading fault chains (expert scenarios)
- Configurable max steps

---

### Step 5: OpenEnv-Compliant API

**File:** `src/server.py` (~340 lines)

A FastAPI REST server that any AI agent can connect to over HTTP.

**Endpoints:**

| Method      | Endpoint                  | Purpose                             |
| ----------- | ------------------------- | ----------------------------------- |
| `GET`       | `/health`                 | Health check                        |
| `GET`       | `/api/v1/tasks`           | List all available tasks            |
| `GET`       | `/api/v1/tasks/{key}`     | Get task details                    |
| `GET`       | `/api/v1/scenarios`       | List all 11 scenarios with metadata |
| `GET`       | `/api/v1/scenarios/{key}` | Get single scenario details         |
| `POST`      | `/api/v1/env/reset`       | Create a new environment instance   |
| `POST`      | `/api/v1/env/{id}/step`   | Execute a command, get reward       |
| `GET`       | `/api/v1/env/{id}/state`  | Debug: full environment state       |
| `GET`       | `/api/v1/env`             | List all active environments        |
| `DELETE`    | `/api/v1/env/{id}`        | Destroy an environment              |
| `WebSocket` | `/ws/env/{id}`            | Live terminal streaming             |
| `POST`      | `/api/v1/arena/run`       | Compare two agents on same scenario |

**How the API works:**

```
1. POST /api/v1/env/reset  {"scenario": "cascading_db_failure"}
   → Returns: env_id, initial observation, task instructions

2. POST /api/v1/env/env_0/step  {"action": "ps"}
   → Returns: command output, reward, score, done flag

3. Repeat step 2 until done=true or max steps reached

4. DELETE /api/v1/env/env_0  (cleanup)
```

---

### Step 6: Agent Interface

**File:** `src/agent.py` (~270 lines)

Two ways to drive the environment:

**AIWorker** (Programmatic — no LLM needed):

```python
from src.agent import AIWorker

agent = AIWorker(scenario="cascading_db_failure")
initial = agent.boot()
result = agent.invoke("ps")                    # run a command
result = agent.invoke("systemctl restart postgres")
print(agent.report())                          # final score + history
```

**LLMAgent** (Optional — requires litellm + API key):

```python
from src.agent import LLMAgent

agent = LLMAgent(model="openai/gpt-4o", api_key="sk-...")
result = agent.solve("cascading_db_failure")   # fully autonomous
# Agent will: observe → think → run command → repeat until solved
```

---

### Step 7: Arena Mode

Compare two agents (or two LLM models) solving the same scenario:

```
POST /api/v1/arena/run
{
  "scenario": "process_recovery",
  "commands_a": ["ps", "systemctl restart postgres", "systemctl status postgres"],
  "commands_b": ["systemctl status postgres", "systemctl restart postgres", "ps"],
  "label_a": "GPT-4",
  "label_b": "Claude"
}
→ Returns: winner, scores, step-by-step history for both
```

---

## Who Will Use This

| User                                | How They Use It                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| **Hackathon judges**                | Hit `/api/v1/env/reset`, run their agent against it, see if it scores 1.0    |
| **AI researchers**                  | Benchmark different LLMs on SRE tasks (GPT-4 vs Claude vs Gemini)            |
| **Companies building AI ops tools** | Test their AI agent's troubleshooting ability before deploying to production |
| **Students learning SRE**           | Step through scenarios manually to learn Linux troubleshooting               |
| **Platform builders**               | Use the scenario engine to create custom failure scenarios for training      |

---

## Use Cases

1. **AI Agent Evaluation** — "Can GPT-4 actually fix a crashed database?" Run it against the `process_recovery` scenario and get an objective 0-1 score.

2. **Benchmarking** — Compare 5 different LLMs on all 11 scenarios. Rank them by score and steps used. Publish results.

3. **Training Data Generation** — Run thousands of episodes, collect command-reward pairs. Use for fine-tuning SRE-specific models.

4. **Chaos Engineering for AI** — Build custom cascading failure scenarios to stress-test AI agents before they touch real infrastructure.

5. **Interactive Learning** — A junior engineer (or AI) works through scenarios from easy to expert, building skills progressively.

---

## Tech Stack

| Component       | Technology                                                |
| --------------- | --------------------------------------------------------- |
| Language        | Python 3.11+                                              |
| API Server      | FastAPI + Uvicorn                                         |
| Data Validation | Pydantic                                                  |
| Live Streaming  | WebSockets                                                |
| Container       | Docker                                                    |
| LLM Integration | LiteLLM (optional)                                        |
| Dependencies    | 4 packages (fastapi, uvicorn, pydantic, python-multipart) |

---

## Quick Start

```bash
# Install
cd meta
pip install -r requirements.txt

# Run the demo
python demo.py

# Start the API server
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000

# Test it
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/scenarios
curl -X POST http://localhost:8000/api/v1/env/reset -H "Content-Type: application/json" -d '{"scenario": "cascading_db_failure"}'
```

---

## Project Structure

```
meta/
├── src/
│   ├── virtual_filesystem.py   # Simulated Linux OS (files, services, disk, network, memory)
│   ├── terminal_emulator.py    # 37-command shell with pipes, redirects, chaining
│   ├── scenarios.py            # 11 composable scenarios with cascading faults
│   ├── tasks.py                # Task interface + legacy compatibility
│   ├── environment.py          # OpenEnv reset/step loop
│   ├── agent.py                # AIWorker + optional LLMAgent
│   ├── server.py               # FastAPI server (14 REST endpoints + WebSocket + Arena)
│   └── __init__.py
├── tests/
│   └── test_environment.py
├── demo.py
├── requirements.txt
├── Dockerfile
├── openenv.yaml
├── README.md                   # Original readme
└── Readmenew.md                # This file
```

---

## What Makes This Different

1. **Cascading failures** — Faults trigger other faults, just like in real production outages. No other AI training environment does this.

2. **Composable scenarios** — Mix and match fault primitives to create any failure scenario. Not limited to pre-built tasks.

3. **37 real Linux commands** — Not a toy. Supports pipes, redirects, variable expansion, path resolution.

4. **Objective grading** — Every scenario is scored 0.0 to 1.0 based on measurable outcomes (is the service running? is the disk below 85%?). No subjective evaluation.

5. **Arena mode** — Side-by-side agent comparison on identical scenarios.

6. **Zero infrastructure** — Runs anywhere Python runs. No VMs, no Docker required, no GPU. Millisecond resets.
