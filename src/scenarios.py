"""
Composable Scenario Engine with Cascading Fault Injection.

Scenarios define:
  - Which services exist and their initial state
  - File overrides / additions
  - Faults to inject (service crash, disk fill, config corruption, etc.)
  - Cascading rules  (fault A triggers fault B when condition C is met)
  - Objectives with staged grading

This is the key differentiator: users can compose failure scenarios from
modular building blocks, and faults cascade just like in real production.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .virtual_filesystem import SystemStore
from .terminal_emulator import Shell


# ======================================================================
#  FAULT PRIMITIVES
# ======================================================================

@dataclass
class Fault:
    """A single injectable fault."""
    name: str
    description: str
    apply_fn: str          # name of a method on FaultInjector
    params: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False


@dataclass
class CascadeRule:
    """When *condition* becomes true, apply *effect* fault."""
    condition_fn: str       # method name on CascadeEngine returning bool
    condition_params: Dict[str, Any] = field(default_factory=dict)
    effect: Fault = field(default_factory=lambda: Fault("noop", "", "noop"))
    triggered: bool = False


@dataclass
class Objective:
    """A single grading objective within a scenario."""
    description: str
    check_fn: str           # method name on ScenarioGrader returning bool
    check_params: Dict[str, Any] = field(default_factory=dict)
    points: float = 0.25
    completed: bool = False


@dataclass
class Scenario:
    """A complete failure scenario definition."""
    name: str
    description: str
    difficulty: str                          # easy / medium / hard / expert
    services: Dict[str, Dict] = field(default_factory=dict)
    file_overrides: Dict[str, Dict] = field(default_factory=dict)
    dir_additions: List[str] = field(default_factory=list)
    faults: List[Fault] = field(default_factory=list)
    cascades: List[CascadeRule] = field(default_factory=list)
    objectives: List[Objective] = field(default_factory=list)
    max_steps: int = 50
    hints: List[str] = field(default_factory=list)

    def guide(self) -> str:
        lines = [
            f"SCENARIO: {self.name} ({self.difficulty})",
            f"{'=' * 60}",
            self.description,
            "",
            "OBJECTIVES:",
        ]
        for i, obj in enumerate(self.objectives, 1):
            lines.append(f"  {i}. {obj.description} ({obj.points:.0%})")
        if self.hints:
            lines.append("\nHINTS:")
            for h in self.hints:
                lines.append(f"  - {h}")
        return "\n".join(lines)


# ======================================================================
#  FAULT INJECTOR — applies faults to the virtual environment
# ======================================================================

class FaultInjector:
    """Applies fault primitives to a SystemStore."""

    def __init__(self, store: SystemStore):
        self.store = store

    def inject(self, fault: Fault) -> None:
        fn = getattr(self, fault.apply_fn, None)
        if fn and not fault.applied:
            fn(**fault.params)
            fault.applied = True

    # ---------- fault implementations ----------

    def crash_service(self, service: str) -> None:
        self.store.svc_stop(service)

    def fill_disk(self, mount: str, fill_mb: int) -> None:
        self.store.disk_use(mount, fill_mb)

    def corrupt_config(self, path: str, corruption: str) -> None:
        ok, content = self.store.fetch(path)
        if ok:
            self.store.write_file(path, corruption)

    def bad_permissions(self, path: str, mode: int) -> None:
        self.store.chmod(path, mode)

    def add_log_flood(self, path: str, line: str, count: int = 1000) -> None:
        flood = "\n".join([line] * count) + "\n"
        self.store.write_file(path, flood, append=True)

    def kill_port(self, port: int) -> None:
        if port in self.store.ports:
            self.store.ports[port]["state"] = "CLOSED"

    def drop_cron(self, index: int) -> None:
        if 0 <= index < len(self.store.cron_entries):
            self.store.cron_entries[index]["enabled"] = False

    def fail_cron(self, index: int) -> None:
        if 0 <= index < len(self.store.cron_entries):
            self.store.cron_entries[index]["last_status"] = "FAILED"

    def memory_pressure(self, used_mb: int) -> None:
        self.store.memory["used_mb"] = used_mb
        self.store.memory["free_mb"] = max(
            0, self.store.memory["total_mb"] - used_mb)
        self.store.memory["available_mb"] = self.store.memory["free_mb"]

    def add_unauthorized_access(self, ip: str, count: int = 20) -> None:
        lines = []
        for i in range(count):
            lines.append(
                f"Mar 30 07:{10+i//60:02d}:{i % 60:02d} sre-lab sshd[{5000+i}]: "
                f"Failed password for root from {ip} port {44000+i} ssh2"
            )
        self.store.write_file("/var/log/auth.log",
                              "\n".join(lines) + "\n", append=True)

    def add_service(self, name: str, pid: int, status: str = "running",
                    port: Optional[int] = None, user: str = "root") -> None:
        self.store.add_service(name, pid, status, port, user)

    def write_file(self, path: str, content: str, permissions: int = 0o644) -> None:
        parent = "/".join(path.split("/")[:-1]) or "/"
        if parent not in self.store.paths:
            self.store.mkdir_p(parent)
        self.store.write_file(path, content)
        if permissions != 0o644:
            self.store.chmod(path, permissions)

    def noop(self) -> None:
        pass


# ======================================================================
#  CASCADE ENGINE — checks conditions and triggers secondary faults
# ======================================================================

class CascadeEngine:
    """Evaluates cascade rules after each step."""

    def __init__(self, store: SystemStore, injector: FaultInjector):
        self.store = store
        self.injector = injector

    def tick(self, rules: List[CascadeRule]) -> List[str]:
        """Evaluate all rules, inject triggered faults. Returns list of triggered fault names."""
        triggered: List[str] = []
        for rule in rules:
            if rule.triggered:
                continue
            fn = getattr(self, rule.condition_fn, None)
            if fn and fn(**rule.condition_params):
                self.injector.inject(rule.effect)
                rule.triggered = True
                triggered.append(rule.effect.name)
        return triggered

    # ---------- condition implementations ----------

    def service_is_dead(self, service: str) -> bool:
        ok, info = self.store.svc_info(service)
        return ok and info["status"] == "dead"

    def service_is_running(self, service: str) -> bool:
        ok, info = self.store.svc_info(service)
        return ok and info["status"] == "running"

    def disk_above_pct(self, mount: str, pct: int) -> bool:
        if mount not in self.store.disk_mounts:
            return False
        d = self.store.disk_mounts[mount]
        used_pct = (d["used_mb"] / d["total_mb"]) * 100 if d["total_mb"] else 0
        return used_pct >= pct

    def disk_below_pct(self, mount: str, pct: int) -> bool:
        return not self.disk_above_pct(mount, pct)

    def memory_above_pct(self, pct: int) -> bool:
        mem = self.store.memory
        used_pct = (mem["used_mb"] / mem["total_mb"]) * \
            100 if mem["total_mb"] else 0
        return used_pct >= pct

    def port_is_closed(self, port: int) -> bool:
        if port not in self.store.ports:
            return True
        return self.store.ports[port]["state"] == "CLOSED"

    def file_contains(self, path: str, pattern: str) -> bool:
        ok, content = self.store.fetch(path)
        return ok and pattern in content

    def always(self) -> bool:
        return True


# ======================================================================
#  SCENARIO GRADER — checks objectives
# ======================================================================

class ScenarioGrader:
    """Evaluates objectives against current environment state."""

    def __init__(self, store: SystemStore, shell: Shell):
        self.store = store
        self.shell = shell

    def evaluate(self, objectives: List[Objective]) -> Tuple[float, Dict]:
        total = 0.0
        meta: Dict[str, Any] = {"completed": [], "pending": []}
        for obj in objectives:
            if obj.completed:
                total += obj.points
                meta["completed"].append(obj.description)
                continue
            fn = getattr(self, obj.check_fn, None)
            if fn and fn(**obj.check_params):
                obj.completed = True
                total += obj.points
                meta["completed"].append(obj.description)
            else:
                meta["pending"].append(obj.description)
        meta["score"] = min(total, 1.0)
        return min(total, 1.0), meta

    # ---------- check implementations ----------

    def service_running(self, service: str) -> bool:
        ok, info = self.store.svc_info(service)
        return ok and info["status"] == "running"

    def service_dead(self, service: str) -> bool:
        ok, info = self.store.svc_info(service)
        return ok and info["status"] == "dead"

    def file_executable(self, path: str) -> bool:
        ok, perms = self.store.perms(path)
        return ok and bool(perms & 0o111)

    def file_contains(self, path: str, text: str) -> bool:
        ok, content = self.store.fetch(path)
        return ok and text in content

    def file_not_contains(self, path: str, text: str) -> bool:
        ok, content = self.store.fetch(path)
        return ok and text not in content

    def file_exists(self, path: str) -> bool:
        return self.store.exists(path)

    def port_listening(self, port: int) -> bool:
        if port not in self.store.ports:
            return False
        return self.store.ports[port]["state"] == "LISTEN"

    def disk_below_pct(self, mount: str, pct: int) -> bool:
        if mount not in self.store.disk_mounts:
            return False
        d = self.store.disk_mounts[mount]
        used_pct = (d["used_mb"] / d["total_mb"]) * 100 if d["total_mb"] else 0
        return used_pct < pct

    def command_was_run(self, pattern: str) -> bool:
        import re
        regex = re.compile(pattern)
        return any(regex.search(cmd) for cmd in self.shell.history())

    def env_var_set(self, key: str, value: str) -> bool:
        return self.store.env_vars.get(key) == value

    def cron_entry_enabled(self, index: int) -> bool:
        entries = self.store.cron_list()
        if 0 <= index < len(entries):
            return entries[index]["enabled"]
        return False

    def all_services_running(self) -> bool:
        for name in self.store.svc_list():
            ok, info = self.store.svc_info(name)
            if ok and info["status"] != "running":
                return False
        return True

    def memory_below_pct(self, pct: int) -> bool:
        mem = self.store.mem_info()
        used_pct = (mem["used_mb"] / mem["total_mb"]) * \
            100 if mem["total_mb"] else 0
        return used_pct < pct


# ======================================================================
#  SCENARIO CATALOG — pre-built scenarios
# ======================================================================

def _scenario_log_analysis() -> Scenario:
    return Scenario(
        name="Log Analysis",
        description=(
            "The application has been experiencing intermittent 500 errors. "
            "Find the exact timestamp of the first 500 Internal Server Error in the app log."
        ),
        difficulty="easy",
        objectives=[
            Objective("Find the 500 error timestamp in /var/log/app.log",
                      "command_was_run", {
                          "pattern": r"grep.*500.*app\.log|cat.*app\.log"},
                      points=1.0),
        ],
        max_steps=50,
        hints=["Check /var/log/app.log", "Use grep to search for '500'"],
    )


def _scenario_permission_repair() -> Scenario:
    return Scenario(
        name="Permission Repair",
        description=(
            "The cleanup.sh script needs to be run but it's not executable. "
            "Fix the permissions so the script can be executed."
        ),
        difficulty="medium",
        faults=[
            Fault("bad_perms", "cleanup.sh is not executable",
                  "bad_permissions",
                  {"path": "/home/user/scripts/cleanup.sh", "mode": 0o644}),
        ],
        objectives=[
            Objective("Make cleanup.sh executable",
                      "file_executable",
                      {"path": "/home/user/scripts/cleanup.sh"},
                      points=1.0),
        ],
        max_steps=50,
        hints=["Use ls -la to check permissions",
               "chmod can set execute bits"],
    )


def _scenario_process_recovery() -> Scenario:
    return Scenario(
        name="Process Recovery",
        description=(
            "The PostgreSQL database service has crashed and is no longer running. "
            "Diagnose the issue and bring the service back online."
        ),
        difficulty="hard",
        faults=[
            Fault("pg_crash", "PostgreSQL is dead",
                  "crash_service", {"service": "postgres"}),
        ],
        objectives=[
            Objective("Identify that postgres is dead",
                      "command_was_run", {
                          "pattern": r"ps|systemctl status postgres"},
                      points=0.2),
            Objective("Restart postgres service",
                      "service_running", {"service": "postgres"},
                      points=0.5),
            Objective("Verify postgres is listening on port 5432",
                      "port_listening", {"port": 5432},
                      points=0.3),
        ],
        max_steps=50,
        hints=["Use ps to see process status",
               "systemctl restart can bring services back"],
    )


def _scenario_cascading_db_failure() -> Scenario:
    """DB crashes → app throws 502s → nginx logs error flood → /var/log fills up."""
    return Scenario(
        name="Cascading Database Failure",
        description=(
            "A cascading failure has occurred: the database crashed, which caused the "
            "application to return 502 errors, which flooded the nginx error log, which "
            "is now filling up the /var/log partition. Find the root cause and fix everything."
        ),
        difficulty="expert",
        faults=[
            Fault("db_crash", "PostgreSQL has crashed",
                  "crash_service", {"service": "postgres"}),
            Fault("log_flood", "Nginx error log flooded with upstream errors",
                  "add_log_flood",
                  {"path": "/var/log/nginx/error.log",
                   "line": "2026/03/30 09:22:20 [error] 1024#0: *99 no live upstreams while connecting to upstream",
                   "count": 5000}),
        ],
        cascades=[
            CascadeRule(
                condition_fn="service_is_dead",
                condition_params={"service": "postgres"},
                effect=Fault("disk_fill", "Log flood fills /var/log partition",
                             "fill_disk", {"mount": "/var/log", "fill_mb": 8500}),
            ),
            CascadeRule(
                condition_fn="disk_above_pct",
                condition_params={"mount": "/var/log", "pct": 95},
                effect=Fault("app_degraded", "App can't write logs, starts failing",
                             "crash_service", {"service": "app"}),
            ),
        ],
        objectives=[
            Objective("Identify that postgres is the root cause",
                      "command_was_run", {
                          "pattern": r"ps|systemctl.*(status|restart).*postgres"},
                      points=0.15),
            Objective("Restart postgres",
                      "service_running", {"service": "postgres"},
                      points=0.25),
            Objective("Restart the app service",
                      "service_running", {"service": "app"},
                      points=0.20),
            Objective("Free up disk space on /var/log (below 90%)",
                      "disk_below_pct", {"mount": "/var/log", "pct": 90},
                      points=0.20),
            Objective("All services running",
                      "all_services_running", {},
                      points=0.20),
        ],
        max_steps=80,
        hints=[
            "Start by checking which services are down (ps, systemctl)",
            "Check disk usage with df -h",
            "Look at /var/log/nginx/error.log for clues",
            "You may need to clean up logs before restarting services",
            "Restart services in dependency order: database first, then app",
        ],
    )


def _scenario_disk_space_crisis() -> Scenario:
    return Scenario(
        name="Disk Space Crisis",
        description=(
            "The /var/log partition is nearly full (>90%). Old log files are consuming "
            "too much space. Free up disk space to prevent service failures."
        ),
        difficulty="medium",
        faults=[
            Fault("disk_full", "Disk nearly full",
                  "fill_disk", {"mount": "/var/log", "fill_mb": 8500}),
            Fault("old_logs", "Huge old debug log in /tmp",
                  "write_file",
                  {"path": "/tmp/old_debug.log",
                   "content": "DEBUG " * 50000 + "\n"}),
        ],
        objectives=[
            Objective("Check disk usage",
                      "command_was_run", {"pattern": r"df|du"},
                      points=0.3),
            Objective("Free disk space on /var/log (below 85%)",
                      "disk_below_pct", {"mount": "/var/log", "pct": 85},
                      points=0.7),
        ],
        max_steps=50,
        hints=["df -h shows disk usage", "du -sh /var/log/* shows per-file usage",
               "rm can remove large files", "Check /tmp for old files too"],
    )


def _scenario_cron_job_failure() -> Scenario:
    return Scenario(
        name="Cron Job Failure",
        description=(
            "The automated backup cron job has been failing repeatedly. "
            "The backup script at /usr/local/bin/backup.sh needs to exist and be executable. "
            "Investigate the cron logs and fix the issue."
        ),
        difficulty="medium",
        faults=[
            Fault("backup_missing", "Backup script doesn't exist at expected path",
                  "noop", {}),
            Fault("cron_failing", "Backup cron job reports FAILED",
                  "fail_cron", {"index": 0}),
        ],
        objectives=[
            Objective("Check cron logs for failures",
                      "command_was_run", {
                          "pattern": r"cat.*cron|grep.*cron|journalctl|crontab"},
                      points=0.2),
            Objective("Create the backup script",
                      "file_exists", {"path": "/usr/local/bin/backup.sh"},
                      points=0.4),
            Objective("Make backup script executable",
                      "file_executable", {"path": "/usr/local/bin/backup.sh"},
                      points=0.4),
        ],
        max_steps=50,
        hints=["Check /var/log/cron.log", "crontab -l shows scheduled jobs",
               "The script path is /usr/local/bin/backup.sh"],
    )


def _scenario_nginx_misconfiguration() -> Scenario:
    return Scenario(
        name="Nginx Misconfiguration",
        description=(
            "After a recent deployment, nginx is returning 502 Bad Gateway errors. "
            "The upstream application server port was changed but nginx config wasn't "
            "updated. Fix the nginx configuration and reload the service."
        ),
        difficulty="hard",
        faults=[
            Fault("wrong_upstream", "Nginx pointing to wrong upstream port",
                  "corrupt_config",
                  {"path": "/etc/nginx/sites-enabled/default",
                   "corruption": (
                       "server {\n    listen 80 default_server;\n    server_name _;\n\n"
                       "    location / {\n        proxy_pass http://127.0.0.1:9999;\n"
                       "        proxy_set_header Host $host;\n"
                       "        proxy_set_header X-Real-IP $remote_addr;\n    }\n}\n"
                   )}),
        ],
        objectives=[
            Objective("Check nginx error logs for upstream errors",
                      "command_was_run", {
                          "pattern": r"cat.*nginx.*error|grep.*nginx|journalctl.*nginx"},
                      points=0.2),
            Objective("Fix nginx config to point to port 8080",
                      "file_contains", {
                          "path": "/etc/nginx/sites-enabled/default", "text": "8080"},
                      points=0.5),
            Objective("Restart nginx",
                      "service_running", {"service": "nginx"},
                      points=0.3),
        ],
        max_steps=50,
        hints=["Check /var/log/nginx/error.log", "The app runs on port 8080",
               "nginx config is in /etc/nginx/sites-enabled/default"],
    )


def _scenario_security_incident() -> Scenario:
    return Scenario(
        name="Security Incident Response",
        description=(
            "An alert has been triggered: there are signs of a brute-force SSH attack "
            "from a suspicious IP address. Investigate the auth logs, identify the "
            "attacker IP, and block it using iptables. Also check if any unauthorized "
            "access was successful."
        ),
        difficulty="hard",
        faults=[
            Fault("brute_force", "Massive SSH brute force from 10.99.99.5",
                  "add_unauthorized_access",
                  {"ip": "10.99.99.5", "count": 50}),
        ],
        objectives=[
            Objective("Examine auth logs",
                      "command_was_run", {
                          "pattern": r"cat.*auth|grep.*auth|tail.*auth"},
                      points=0.3),
            Objective("Identify attacker IP (10.99.99.5)",
                      "command_was_run", {
                          "pattern": r"grep.*10\.99\.99\.5|grep.*Failed"},
                      points=0.3),
            Objective("Verify no successful root login from attacker",
                      "command_was_run", {
                          "pattern": r"grep.*Accepted.*10\.99|grep -v.*Failed"},
                      points=0.4),
        ],
        max_steps=50,
        hints=["Check /var/log/auth.log", "Look for 'Failed password' entries",
               "grep can filter by IP address"],
    )


def _scenario_memory_leak() -> Scenario:
    return Scenario(
        name="Memory Leak Investigation",
        description=(
            "The server is running critically low on memory. A service appears to be "
            "leaking memory. Identify which service is consuming excessive memory, "
            "restart it, and verify memory is freed."
        ),
        difficulty="hard",
        faults=[
            Fault("mem_pressure", "System under memory pressure",
                  "memory_pressure", {"used_mb": 7800}),
        ],
        objectives=[
            Objective("Check system memory",
                      "command_was_run", {"pattern": r"free|top|cat.*meminfo"},
                      points=0.2),
            Objective("Identify the memory-hungry process",
                      "command_was_run", {"pattern": r"top|ps.*aux"},
                      points=0.3),
            Objective("Restart the problematic service",
                      "command_was_run", {
                          "pattern": r"systemctl.*restart|kill"},
                      points=0.3),
            Objective("Verify memory is freed (below 90%)",
                      "memory_below_pct", {"pct": 90},
                      points=0.2),
        ],
        max_steps=50,
        hints=["free -m shows memory usage", "top shows per-process memory",
               "Restarting a service releases its memory"],
    )


def _scenario_network_troubleshooting() -> Scenario:
    return Scenario(
        name="Network Connectivity Issue",
        description=(
            "Users are reporting that the application is unreachable. The web server "
            "appears to be running but connections are not going through. Diagnose "
            "the network path: DNS, ports, firewall, and service health."
        ),
        difficulty="hard",
        faults=[
            Fault("port_blocked", "App port closed in iptables but service running",
                  "kill_port", {"port": 8080}),
        ],
        objectives=[
            Objective("Check if services are running",
                      "command_was_run", {"pattern": r"ps|systemctl.*status"},
                      points=0.2),
            Objective("Check network ports",
                      "command_was_run", {"pattern": r"netstat|ss|curl"},
                      points=0.3),
            Objective("Identify the blocked port",
                      "command_was_run", {
                          "pattern": r"iptables|netstat|curl.*8080"},
                      points=0.3),
            Objective("Restart the app service to re-bind port",
                      "port_listening", {"port": 8080},
                      points=0.2),
        ],
        max_steps=50,
        hints=["netstat -tlnp shows listening ports", "curl localhost:8080 tests connectivity",
               "iptables -L shows firewall rules", "Restarting the service may re-bind the port"],
    )


def _scenario_full_incident() -> Scenario:
    """The ultimate challenge: multiple simultaneous failures."""
    return Scenario(
        name="Full Incident Response",
        description=(
            "CRITICAL ALERT: Multiple systems are failing simultaneously.\n"
            "- Database is down\n"
            "- Disk is nearly full on /var/log\n"
            "- Backup cron jobs are failing\n"
            "- There are signs of a brute-force attack\n\n"
            "Triage, prioritize, and resolve all issues. Restore all services, "
            "free disk space, and secure the system."
        ),
        difficulty="expert",
        faults=[
            Fault("db_down", "PostgreSQL crashed",
                  "crash_service", {"service": "postgres"}),
            Fault("disk_full", "Log partition nearly full",
                  "fill_disk", {"mount": "/var/log", "fill_mb": 8500}),
            Fault("cron_fail", "Backup cron failing",
                  "fail_cron", {"index": 0}),
            Fault("brute_ssh", "SSH brute force attack",
                  "add_unauthorized_access", {"ip": "10.99.99.5", "count": 30}),
            Fault("mem_pressure", "High memory usage",
                  "memory_pressure", {"used_mb": 7500}),
        ],
        cascades=[
            CascadeRule(
                condition_fn="service_is_dead",
                condition_params={"service": "postgres"},
                effect=Fault("app_errors", "App throwing errors due to DB",
                             "add_log_flood",
                             {"path": "/var/log/app.log",
                              "line": "2026-03-30T09:30:00.000Z [ERROR] Database connection refused",
                              "count": 500}),
            ),
        ],
        objectives=[
            Objective("Triage: check overall system health",
                      "command_was_run", {"pattern": r"ps|top|df|free"},
                      points=0.10),
            Objective("Restart postgres",
                      "service_running", {"service": "postgres"},
                      points=0.20),
            Objective("Free up disk space (below 85%)",
                      "disk_below_pct", {"mount": "/var/log", "pct": 85},
                      points=0.20),
            Objective("Investigate security incident",
                      "command_was_run", {"pattern": r"grep.*auth|cat.*auth"},
                      points=0.15),
            Objective("Ensure all services healthy",
                      "all_services_running", {},
                      points=0.20),
            Objective("Verify application is responding",
                      "command_was_run", {
                          "pattern": r"curl.*localhost|curl.*8080"},
                      points=0.15),
        ],
        max_steps=100,
        hints=[
            "Start with triage: ps, df -h, free -m",
            "Fix the most critical issues first (database, disk)",
            "Check /var/log/auth.log for the attack",
            "Clean old logs to free disk",
            "Restart services in dependency order",
        ],
    )


# ======================================================================
#  CATALOG REGISTRY
# ======================================================================

SCENARIO_CATALOG: Dict[str, Callable[[], Scenario]] = {
    "log_analysis":           _scenario_log_analysis,
    "permission_repair":      _scenario_permission_repair,
    "process_recovery":       _scenario_process_recovery,
    "cascading_db_failure":   _scenario_cascading_db_failure,
    "disk_space_crisis":      _scenario_disk_space_crisis,
    "cron_job_failure":       _scenario_cron_job_failure,
    "nginx_misconfiguration": _scenario_nginx_misconfiguration,
    "security_incident":      _scenario_security_incident,
    "memory_leak":            _scenario_memory_leak,
    "network_troubleshooting": _scenario_network_troubleshooting,
    "full_incident":          _scenario_full_incident,
}

# backward-compat mapping: difficulty -> default scenario
DIFFICULTY_MAP: Dict[str, str] = {
    "easy":   "log_analysis",
    "medium": "permission_repair",
    "hard":   "process_recovery",
    "expert": "cascading_db_failure",
}


def list_scenarios() -> Dict[str, Dict]:
    """Return metadata for all registered scenarios."""
    result = {}
    for key, factory in SCENARIO_CATALOG.items():
        s = factory()
        result[key] = {
            "name": s.name,
            "difficulty": s.difficulty,
            "description": s.description,
            "objectives_count": len(s.objectives),
            "max_steps": s.max_steps,
        }
    return result


def load_scenario(key: str) -> Scenario:
    """Load and return a fresh copy of a scenario."""
    factory = SCENARIO_CATALOG.get(key)
    if not factory:
        raise ValueError(
            f"Unknown scenario: '{key}'. Available: {list(SCENARIO_CATALOG.keys())}")
    return factory()
