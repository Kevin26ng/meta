import copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


class SystemStore:

    def __init__(self):
        self.index: Dict[str, Dict] = {}
        self.paths: set = set()
        self.svc: Dict[str, Dict] = {}
        self.disk_mounts: Dict[str, Dict] = {}
        self.ports: Dict[int, Dict] = {}
        self.memory: Dict[str, int] = {}
        self.cron_entries: List[Dict] = []
        self.users: Dict[str, Dict] = {}
        self.groups: Dict[str, List[str]] = {}
        self.env_vars: Dict[str, str] = {}
        self.hostname: str = "sre-lab"
        self.uptime_secs: int = 86400
        self.load_avg: Tuple[float, float, float] = (0.45, 0.32, 0.28)
        self.iptables_rules: List[str] = []
        self.dns_records: Dict[str, str] = {}
        self._initial_state: Optional[Dict] = None
        self._setup()
        self._initial_state = self._deep_snapshot()

    # ------------------------------------------------------------------ setup
    def _setup(self):
        self._init_dirs()
        self._init_files()
        self._init_services()
        self._init_disk()
        self._init_network()
        self._init_memory()
        self._init_cron()
        self._init_users()
        self._init_env()
        self._init_firewall()
        self._init_dns()

    def _init_dirs(self):
        self.paths.update([
            "/", "/bin", "/usr", "/usr/bin", "/opt", "/tmp",
            "/var", "/var/log", "/var/log/nginx", "/var/run",
            "/var/spool", "/var/spool/cron",
            "/etc", "/etc/nginx", "/etc/nginx/sites-enabled",
            "/etc/systemd", "/etc/systemd/system",
            "/etc/cron.d",
            "/proc",
            "/home", "/home/user", "/home/user/scripts",
            "/home/user/.ssh",
        ])

    def _init_files(self):
        # ---- application logs ----
        self._add_content("/var/log/app.log", [
            "2026-03-30T08:15:23.123Z [INFO] Application started",
            "2026-03-30T08:15:45.456Z [DEBUG] Initializing database connection",
            "2026-03-30T08:16:01.789Z [INFO] Database pool created size=10",
            "2026-03-30T08:30:00.000Z [INFO] Health check passed",
            "2026-03-30T09:00:00.000Z [INFO] Health check passed",
            "2026-03-30T09:22:17.789Z [ERROR] Database connection timeout",
            "2026-03-30T09:22:18.012Z [ERROR] Failed to fetch user data",
            "2026-03-30T09:22:19.234Z [ERROR] 500 Internal Server Error",
            "2026-03-30T09:22:20.100Z [WARN] Connection pool exhausted",
            "2026-03-30T09:22:21.345Z [ERROR] 500 Internal Server Error",
            "2026-03-30T09:22:22.567Z [ERROR] 500 Internal Server Error",
            "2026-03-30T09:23:01.567Z [INFO] Attempting reconnection",
            "2026-03-30T09:23:15.890Z [INFO] Connection restored",
            "2026-03-30T09:30:00.000Z [INFO] Health check passed",
            "2026-03-30T10:00:00.000Z [INFO] Health check passed",
        ], permissions=0o644)

        # ---- nginx logs ----
        self._add_content("/var/log/nginx/access.log", [
            '192.168.1.10 - - [30/Mar/2026:08:15:30 +0000] "GET / HTTP/1.1" 200 612',
            '192.168.1.10 - - [30/Mar/2026:08:16:02 +0000] "GET /api/health HTTP/1.1" 200 15',
            '192.168.1.15 - - [30/Mar/2026:09:22:18 +0000] "GET /api/users HTTP/1.1" 502 182',
            '192.168.1.15 - - [30/Mar/2026:09:22:19 +0000] "GET /api/users HTTP/1.1" 502 182',
            '10.0.0.5 - - [30/Mar/2026:09:22:20 +0000] "POST /api/data HTTP/1.1" 502 182',
            '192.168.1.10 - - [30/Mar/2026:09:23:16 +0000] "GET /api/health HTTP/1.1" 200 15',
            '192.168.1.20 - - [30/Mar/2026:09:30:00 +0000] "GET / HTTP/1.1" 200 612',
        ], permissions=0o644)

        self._add_content("/var/log/nginx/error.log", [
            "2026/03/30 09:22:18 [error] 1024#0: *15 connect() failed (111: Connection refused) while connecting to upstream",
            "2026/03/30 09:22:19 [error] 1024#0: *16 connect() failed (111: Connection refused) while connecting to upstream",
            "2026/03/30 09:22:20 [error] 1024#0: *17 no live upstreams while connecting to upstream",
        ], permissions=0o644)

        # ---- syslog ----
        self._add_content("/var/log/syslog", [
            "Mar 30 08:15:00 sre-lab systemd[1]: Started Application Server.",
            "Mar 30 08:15:01 sre-lab systemd[1]: Started nginx - high performance web server.",
            "Mar 30 08:15:02 sre-lab systemd[1]: Started Redis Server.",
            "Mar 30 09:22:15 sre-lab systemd[1]: postgres.service: Main process exited, code=killed, status=9/KILL",
            "Mar 30 09:22:15 sre-lab systemd[1]: postgres.service: Failed with result 'signal'.",
            "Mar 30 09:22:16 sre-lab kernel: [42135.123456] Out of memory: Killed process 5432 (postgres)",
            "Mar 30 10:00:00 sre-lab CRON[9876]: (root) CMD (/usr/bin/logrotate /etc/logrotate.conf)",
            "Mar 30 10:15:00 sre-lab systemd[1]: Starting Daily apt activities...",
        ], permissions=0o644)

        # ---- auth log ----
        self._add_content("/var/log/auth.log", [
            "Mar 30 06:15:00 sre-lab sshd[4321]: Accepted publickey for user from 192.168.1.10 port 52431",
            "Mar 30 07:02:11 sre-lab sshd[4400]: Failed password for root from 10.99.99.5 port 44312 ssh2",
            "Mar 30 07:02:14 sre-lab sshd[4400]: Failed password for root from 10.99.99.5 port 44312 ssh2",
            "Mar 30 07:02:17 sre-lab sshd[4400]: Failed password for root from 10.99.99.5 port 44312 ssh2",
            "Mar 30 07:02:20 sre-lab sshd[4401]: Failed password for admin from 10.99.99.5 port 44315 ssh2",
            "Mar 30 07:02:23 sre-lab sshd[4401]: Failed password for admin from 10.99.99.5 port 44315 ssh2",
            "Mar 30 07:02:26 sre-lab sshd[4402]: Invalid user test from 10.99.99.5 port 44320",
            "Mar 30 07:03:00 sre-lab sshd[4402]: Connection closed by 10.99.99.5 port 44320 [preauth]",
            "Mar 30 08:00:00 sre-lab sudo: user : TTY=pts/0 ; PWD=/home/user ; USER=root ; COMMAND=/bin/systemctl restart app",
        ], permissions=0o640)

        # ---- cron log ----
        self._add_content("/var/log/cron.log", [
            "Mar 30 00:00:01 sre-lab CRON[8001]: (root) CMD (/usr/local/bin/backup.sh)",
            "Mar 30 00:00:01 sre-lab CRON[8001]: (root) FAILED (exit code 1)",
            "Mar 30 06:00:01 sre-lab CRON[8100]: (root) CMD (/usr/local/bin/backup.sh)",
            "Mar 30 06:00:01 sre-lab CRON[8100]: (root) FAILED (exit code 1)",
            "Mar 30 10:00:00 sre-lab CRON[9876]: (root) CMD (/usr/bin/logrotate /etc/logrotate.conf)",
            "Mar 30 12:00:01 sre-lab CRON[8200]: (root) CMD (/usr/local/bin/backup.sh)",
            "Mar 30 12:00:01 sre-lab CRON[8200]: (root) FAILED (exit code 1)",
        ], permissions=0o644)

        # ---- config files ----
        self._store_file("/etc/app.conf",
                         "[database]\nhost=localhost\nport=5432\npool_size=10\nmax_overflow=5\ntimeout=30\n\n"
                         "[server]\nhost=0.0.0.0\nport=8080\nworkers=4\ndebug=false\n\n"
                         "[logging]\nlevel=INFO\nfile=/var/log/app.log\n",
                         permissions=0o640)

        self._store_file("/etc/nginx/nginx.conf",
                         "user www-data;\nworker_processes auto;\npid /var/run/nginx.pid;\n\n"
                         "events {\n    worker_connections 768;\n}\n\n"
                         "http {\n    sendfile on;\n    tcp_nopush on;\n    types_hash_max_size 2048;\n\n"
                         "    include /etc/nginx/sites-enabled/*;\n\n"
                         "    access_log /var/log/nginx/access.log;\n    error_log /var/log/nginx/error.log;\n}\n",
                         permissions=0o644)

        self._store_file("/etc/nginx/sites-enabled/default",
                         "server {\n    listen 80 default_server;\n    server_name _;\n\n"
                         "    location / {\n        proxy_pass http://127.0.0.1:8080;\n"
                         "        proxy_set_header Host $host;\n"
                         "        proxy_set_header X-Real-IP $remote_addr;\n    }\n\n"
                         "    location /static {\n        alias /opt/app/static;\n    }\n}\n",
                         permissions=0o644)

        self._store_file("/etc/hosts",
                         "127.0.0.1\tlocalhost\n127.0.1.1\tsre-lab\n"
                         "192.168.1.100\tdb-primary\n192.168.1.101\tdb-replica\n"
                         "192.168.1.200\tcache-01\n",
                         permissions=0o644)

        self._store_file("/etc/resolv.conf",
                         "nameserver 8.8.8.8\nnameserver 8.8.4.4\nsearch example.com\n",
                         permissions=0o644)

        self._store_file("/etc/passwd",
                         "root:x:0:0:root:/root:/bin/bash\n"
                         "www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin\n"
                         "postgres:x:108:113:PostgreSQL administrator,,,:/var/lib/postgresql:/bin/bash\n"
                         "redis:x:109:114::/var/lib/redis:/usr/sbin/nologin\n"
                         "user:x:1000:1000:SRE User:/home/user:/bin/bash\n"
                         "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n",
                         permissions=0o644)

        self._store_file("/etc/crontab",
                         "SHELL=/bin/bash\nPATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin\n\n"
                         "# m h dom mon dow user command\n"
                         "0 */6 * * * root /usr/local/bin/backup.sh\n"
                         "0 0 * * * root /usr/bin/logrotate /etc/logrotate.conf\n"
                         "*/5 * * * * root /usr/local/bin/health-check.sh\n",
                         permissions=0o644)

        # ---- systemd service units ----
        self._store_file("/etc/systemd/system/app.service",
                         "[Unit]\nDescription=Application Server\nAfter=network.target postgres.service\nRequires=postgres.service\n\n"
                         "[Service]\nType=simple\nUser=user\nWorkingDirectory=/opt/app\n"
                         "ExecStart=/usr/bin/python3 /opt/app/server.py\nRestart=on-failure\nRestartSec=5\n\n"
                         "[Install]\nWantedBy=multi-user.target\n",
                         permissions=0o644)

        self._store_file("/etc/systemd/system/postgres.service",
                         "[Unit]\nDescription=PostgreSQL Database Server\nAfter=network.target\n\n"
                         "[Service]\nType=forking\nUser=postgres\n"
                         "ExecStart=/usr/lib/postgresql/14/bin/pg_ctl start -D /var/lib/postgresql/14/main\n"
                         "ExecStop=/usr/lib/postgresql/14/bin/pg_ctl stop -D /var/lib/postgresql/14/main\n"
                         "ExecReload=/bin/kill -HUP $MAINPID\nRestart=on-failure\n\n"
                         "[Install]\nWantedBy=multi-user.target\n",
                         permissions=0o644)

        # ---- user scripts ----
        self._store_file("/home/user/scripts/cleanup.sh",
                         "#!/bin/bash\necho 'Running cleanup...'\nrm -rf /tmp/*.log\necho 'Cleanup complete'",
                         permissions=0o644)

        self._store_file("/home/user/scripts/backup.sh",
                         "#!/bin/bash\nset -e\nBACKUP_DIR=/opt/backups\n"
                         "pg_dump -U postgres appdb > $BACKUP_DIR/db_$(date +%Y%m%d).sql\n"
                         "tar czf $BACKUP_DIR/files_$(date +%Y%m%d).tar.gz /opt/app/data/\n"
                         "echo 'Backup completed successfully'",
                         permissions=0o644)

        self._store_file("/home/user/scripts/deploy.sh",
                         "#!/bin/bash\nset -e\necho 'Pulling latest code...'\ncd /opt/app\n"
                         "git pull origin main\npip install -r requirements.txt\n"
                         "systemctl restart app\necho 'Deploy complete'",
                         permissions=0o755)

        self._store_file("/home/user/scripts/monitor.sh",
                         "#!/bin/bash\necho '=== System Health ==='\nuptime\necho ''\n"
                         "echo '=== Disk Usage ==='\ndf -h\necho ''\n"
                         "echo '=== Memory ==='\nfree -m\necho ''\n"
                         "echo '=== Services ==='\nsystemctl status nginx postgres redis app\n",
                         permissions=0o755)

        self._store_file("/home/user/scripts/health-check.sh",
                         "#!/bin/bash\ncurl -sf http://localhost:8080/health > /dev/null\n"
                         "if [ $? -ne 0 ]; then\n  echo 'Health check FAILED' >> /var/log/app.log\n"
                         "  systemctl restart app\nfi\n",
                         permissions=0o755)

        self._store_file("/home/user/.bashrc",
                         "export PATH=$PATH:/home/user/scripts\nexport EDITOR=vim\n"
                         "alias ll='ls -la'\nalias gs='git status'\n",
                         permissions=0o644)

        self._store_file("/home/user/.ssh/authorized_keys",
                         "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC7... user@workstation\n",
                         permissions=0o600)

        # ---- proc pseudo-files ----
        self._store_file("/proc/meminfo",
                         "MemTotal:        8192000 kB\nMemFree:          512000 kB\n"
                         "MemAvailable:    1024000 kB\nBuffers:          256000 kB\n"
                         "Cached:          1536000 kB\nSwapTotal:       2048000 kB\n"
                         "SwapFree:        1800000 kB\n",
                         permissions=0o444)

        self._store_file("/proc/cpuinfo",
                         "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz\n"
                         "cpu MHz\t\t: 2300.000\ncache size\t: 46080 KB\ncpu cores\t: 2\n\n"
                         "processor\t: 1\nmodel name\t: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz\n"
                         "cpu MHz\t\t: 2300.000\ncache size\t: 46080 KB\ncpu cores\t: 2\n",
                         permissions=0o444)

        self._store_file(
            "/proc/loadavg", "0.45 0.32 0.28 1/120 9876\n", permissions=0o444)

        # ---- PID files ----
        self._store_file("/var/run/nginx.pid", "1024\n", permissions=0o644)

        # ---- tmp ----
        self._store_file("/tmp/app.pid", "8080\n", permissions=0o644)
        self._store_file("/tmp/debug.log",
                         "DEBUG 2026-03-30 some old debug output\nDEBUG 2026-03-30 leftover temp data\n",
                         permissions=0o644)

    def _init_services(self):
        self.svc = {
            "nginx":    {"pid": 1024, "status": "running", "restartable": True,
                         "cpu": 0.3, "mem_mb": 64,  "port": 80,   "user": "www-data"},
            "postgres": {"pid": 5432, "status": "dead",    "restartable": True,
                         "cpu": 0.0, "mem_mb": 0,   "port": 5432, "user": "postgres"},
            "redis":    {"pid": 6379, "status": "running", "restartable": True,
                         "cpu": 0.1, "mem_mb": 128, "port": 6379, "user": "redis"},
            "app":      {"pid": 8080, "status": "running", "restartable": True,
                         "cpu": 2.5, "mem_mb": 256, "port": 8080, "user": "user"},
            "cron":     {"pid": 500,  "status": "running", "restartable": True,
                         "cpu": 0.0, "mem_mb": 4,   "port": None, "user": "root"},
            "sshd":     {"pid": 800,  "status": "running", "restartable": True,
                         "cpu": 0.0, "mem_mb": 8,   "port": 22,   "user": "root"},
        }

    def _init_disk(self):
        self.disk_mounts = {
            "/":        {"device": "/dev/sda1", "fstype": "ext4",
                         "total_mb": 51200, "used_mb": 34816, "available_mb": 16384},
            "/boot":    {"device": "/dev/sda2", "fstype": "ext2",
                         "total_mb": 512,   "used_mb": 128,   "available_mb": 384},
            "/var/log": {"device": "/dev/sda3", "fstype": "ext4",
                         "total_mb": 10240, "used_mb": 8704,  "available_mb": 1536},
            "/tmp":     {"device": "tmpfs",     "fstype": "tmpfs",
                         "total_mb": 4096,  "used_mb": 64,    "available_mb": 4032},
        }

    def _init_network(self):
        self.ports = {
            22:   {"service": "sshd",     "state": "LISTEN",      "proto": "tcp", "addr": "0.0.0.0"},
            80:   {"service": "nginx",    "state": "LISTEN",      "proto": "tcp", "addr": "0.0.0.0"},
            5432: {"service": "postgres", "state": "CLOSED",      "proto": "tcp", "addr": "127.0.0.1"},
            6379: {"service": "redis",    "state": "LISTEN",      "proto": "tcp", "addr": "127.0.0.1"},
            8080: {"service": "app",      "state": "LISTEN",      "proto": "tcp", "addr": "127.0.0.1"},
            443:  {"service": "nginx",    "state": "LISTEN",      "proto": "tcp", "addr": "0.0.0.0"},
        }

    def _init_memory(self):
        self.memory = {
            "total_mb":     8192,
            "used_mb":      6656,
            "free_mb":      512,
            "available_mb": 1024,
            "buffers_mb":   256,
            "cached_mb":    1536,
            "swap_total_mb": 2048,
            "swap_used_mb":  248,
            "swap_free_mb": 1800,
        }

    def _init_cron(self):
        self.cron_entries = [
            {"schedule": "0 */6 * * *",  "user": "root", "command": "/usr/local/bin/backup.sh",
             "enabled": True, "last_run": "2026-03-30T12:00:01", "last_status": "FAILED"},
            {"schedule": "0 0 * * *",    "user": "root", "command": "/usr/bin/logrotate /etc/logrotate.conf",
             "enabled": True, "last_run": "2026-03-30T00:00:00", "last_status": "OK"},
            {"schedule": "*/5 * * * *",  "user": "root", "command": "/usr/local/bin/health-check.sh",
             "enabled": True, "last_run": "2026-03-30T10:00:00", "last_status": "OK"},
        ]

    def _init_users(self):
        self.users = {
            "root":     {"uid": 0,    "gid": 0,   "home": "/root",          "shell": "/bin/bash"},
            "www-data": {"uid": 33,   "gid": 33,  "home": "/var/www",       "shell": "/usr/sbin/nologin"},
            "postgres": {"uid": 108,  "gid": 113, "home": "/var/lib/postgresql", "shell": "/bin/bash"},
            "redis":    {"uid": 109,  "gid": 114, "home": "/var/lib/redis", "shell": "/usr/sbin/nologin"},
            "user":     {"uid": 1000, "gid": 1000, "home": "/home/user",     "shell": "/bin/bash"},
            "nobody":   {"uid": 65534, "gid": 65534, "home": "/nonexistent",  "shell": "/usr/sbin/nologin"},
        }
        self.groups = {
            "root": ["root"], "www-data": ["www-data"], "user": ["user"],
            "sudo": ["user"], "docker": ["user"], "adm": ["user", "root"],
        }

    def _init_env(self):
        self.env_vars = {
            "HOME": "/home/user", "USER": "user", "SHELL": "/bin/bash",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "en_US.UTF-8", "TERM": "xterm-256color",
            "HOSTNAME": self.hostname,
            "DATABASE_URL": "postgresql://localhost:5432/appdb",
            "REDIS_URL": "redis://localhost:6379/0",
            "APP_ENV": "production",
            "LOG_LEVEL": "INFO",
        }

    def _init_firewall(self):
        self.iptables_rules = [
            "Chain INPUT (policy DROP)",
            "  ACCEPT  all  --  lo  lo  0.0.0.0/0  0.0.0.0/0",
            "  ACCEPT  all  --  *   *   0.0.0.0/0  0.0.0.0/0  state RELATED,ESTABLISHED",
            "  ACCEPT  tcp  --  *   *   0.0.0.0/0  0.0.0.0/0  tcp dpt:22",
            "  ACCEPT  tcp  --  *   *   0.0.0.0/0  0.0.0.0/0  tcp dpt:80",
            "  ACCEPT  tcp  --  *   *   0.0.0.0/0  0.0.0.0/0  tcp dpt:443",
            "  DROP    tcp  --  *   *   0.0.0.0/0  0.0.0.0/0  tcp dpt:5432",
            "",
            "Chain FORWARD (policy DROP)",
            "",
            "Chain OUTPUT (policy ACCEPT)",
        ]

    def _init_dns(self):
        self.dns_records = {
            "localhost":     "127.0.0.1",
            "sre-lab":       "127.0.1.1",
            "db-primary":    "192.168.1.100",
            "db-replica":    "192.168.1.101",
            "cache-01":      "192.168.1.200",
            "api.example.com": "93.184.216.34",
        }

    # ----------------------------------------------------------- file helpers
    def _store_file(self, path: str, content: str, permissions: int = 0o644):
        self.index[path] = {
            "content": content,
            "permissions": permissions,
            "owner": "user",
            "group": "user",
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat(),
            "is_executable": bool(permissions & 0o111),
        }

    def _add_content(self, path: str, lines: List[str], permissions: int = 0o644):
        content = "\n".join(lines) + "\n"
        self._store_file(path, content, permissions)

    # ----------------------------------------------------------- file reads
    def fetch(self, path: str) -> Tuple[bool, str]:
        if path not in self.index:
            return False, f"cat: {path}: No such file or directory"
        return True, self.index[path]["content"]

    def entries(self, path: str) -> Tuple[bool, List[str]]:
        norm = path.rstrip("/") if path != "/" else ""
        is_dir = path in self.paths
        if not is_dir:
            return False, f"ls: cannot access '{path}': No such file or directory"

        items = set()
        prefix = norm + "/"
        for fpath in self.index:
            if fpath.startswith(prefix):
                rel = fpath[len(prefix):]
                if "/" not in rel:
                    items.add(rel)
        for directory in self.paths:
            if directory != path and directory.startswith(prefix):
                rel = directory[len(prefix):]
                if "/" not in rel:
                    items.add(rel + "/")
        return True, sorted(items)

    def exists(self, path: str) -> bool:
        return path in self.index

    def dir_exists(self, path: str) -> bool:
        return path in self.paths

    def perms(self, path: str) -> Tuple[bool, int]:
        if path not in self.index:
            return False, 0
        return True, self.index[path]["permissions"]

    def info(self, path: str) -> Tuple[bool, Dict]:
        if path not in self.index:
            return False, {}
        data = self.index[path].copy()
        data["path"] = path
        data["size"] = len(data["content"])
        return True, data

    # ----------------------------------------------------------- file writes
    def chmod(self, path: str, permissions: int) -> Tuple[bool, str]:
        if path not in self.index:
            return False, f"chmod: cannot access '{path}': No such file or directory"
        self.index[path]["permissions"] = permissions
        self.index[path]["is_executable"] = bool(permissions & 0o111)
        self.index[path]["modified_at"] = datetime.now().isoformat()
        return True, ""

    def write_file(self, path: str, content: str, append: bool = False) -> Tuple[bool, str]:
        if path in self.index:
            if append:
                self.index[path]["content"] += content
            else:
                self.index[path]["content"] = content
            self.index[path]["modified_at"] = datetime.now().isoformat()
        else:
            parent = "/".join(path.split("/")[:-1]) or "/"
            if parent not in self.paths:
                return False, f"No such directory: {parent}"
            self._store_file(path, content, permissions=0o644)
        return True, ""

    def touch(self, path: str) -> Tuple[bool, str]:
        if path in self.index:
            self.index[path]["modified_at"] = datetime.now().isoformat()
            return True, ""
        return self.write_file(path, "")

    def mkdir(self, path: str) -> Tuple[bool, str]:
        if path in self.paths:
            return False, f"mkdir: cannot create directory '{path}': File exists"
        parent = "/".join(path.split("/")[:-1]) or "/"
        if parent not in self.paths:
            return False, f"mkdir: cannot create directory '{path}': No such file or directory"
        self.paths.add(path)
        return True, ""

    def mkdir_p(self, path: str) -> Tuple[bool, str]:
        parts = path.strip("/").split("/")
        current = ""
        for part in parts:
            current += "/" + part
            if current not in self.paths:
                self.paths.add(current)
        return True, ""

    def cp(self, src: str, dst: str) -> Tuple[bool, str]:
        if src not in self.index:
            return False, f"cp: cannot stat '{src}': No such file or directory"
        dst_parent = "/".join(dst.split("/")[:-1]) or "/"
        if dst_parent not in self.paths:
            return False, f"cp: cannot create '{dst}': No such file or directory"
        self.index[dst] = copy.deepcopy(self.index[src])
        self.index[dst]["modified_at"] = datetime.now().isoformat()
        return True, ""

    def mv(self, src: str, dst: str) -> Tuple[bool, str]:
        ok, err = self.cp(src, dst)
        if not ok:
            return False, err
        del self.index[src]
        return True, ""

    def rm(self, path: str) -> Tuple[bool, str]:
        if path not in self.index:
            return False, f"rm: cannot remove '{path}': No such file or directory"
        del self.index[path]
        return True, ""

    # ----------------------------------------------------------- services
    def svc_info(self, name: str) -> Tuple[bool, Dict]:
        if name not in self.svc:
            return False, {}
        return True, self.svc[name].copy()

    def svc_list(self) -> List[str]:
        return list(self.svc.keys())

    def svc_stop(self, name: str) -> Tuple[bool, str]:
        if name not in self.svc:
            return False, f"Process '{name}' not found"
        if self.svc[name]["status"] == "dead":
            return False, f"Process '{name}' is already dead"
        self.svc[name]["status"] = "dead"
        self.svc[name]["cpu"] = 0.0
        self.svc[name]["mem_mb"] = 0
        port = self.svc[name].get("port")
        if port and port in self.ports:
            self.ports[port]["state"] = "CLOSED"
        return True, ""

    def svc_start(self, name: str) -> Tuple[bool, str]:
        if name not in self.svc:
            return False, f"Process '{name}' not found"
        if not self.svc[name]["restartable"]:
            return False, f"Process '{name}' cannot be restarted"
        self.svc[name]["status"] = "running"
        self.svc[name]["cpu"] = 0.5
        self.svc[name]["mem_mb"] = max(self.svc[name].get("mem_mb", 0), 32)
        port = self.svc[name].get("port")
        if port and port in self.ports:
            self.ports[port]["state"] = "LISTEN"
        return True, ""

    def add_service(self, name: str, pid: int, status: str = "running",
                    port: Optional[int] = None, user: str = "root",
                    cpu: float = 0.0, mem_mb: int = 32) -> None:
        self.svc[name] = {
            "pid": pid, "status": status, "restartable": True,
            "cpu": cpu, "mem_mb": mem_mb, "port": port, "user": user,
        }
        if port:
            self.ports[port] = {
                "service": name,
                "state": "LISTEN" if status == "running" else "CLOSED",
                "proto": "tcp", "addr": "0.0.0.0",
            }

    # ----------------------------------------------------------- disk
    def disk_usage(self) -> Dict[str, Dict]:
        return copy.deepcopy(self.disk_mounts)

    def disk_use(self, mount: str, amount_mb: int) -> Tuple[bool, str]:
        if mount not in self.disk_mounts:
            return False, f"Mount '{mount}' not found"
        d = self.disk_mounts[mount]
        d["used_mb"] += amount_mb
        d["available_mb"] = max(0, d["total_mb"] - d["used_mb"])
        return True, ""

    def disk_free(self, mount: str, amount_mb: int) -> Tuple[bool, str]:
        if mount not in self.disk_mounts:
            return False, f"Mount '{mount}' not found"
        d = self.disk_mounts[mount]
        d["used_mb"] = max(0, d["used_mb"] - amount_mb)
        d["available_mb"] = d["total_mb"] - d["used_mb"]
        return True, ""

    # ----------------------------------------------------------- network
    def port_info(self, port: int) -> Tuple[bool, Dict]:
        if port not in self.ports:
            return False, {}
        return True, self.ports[port].copy()

    def all_ports(self) -> Dict[int, Dict]:
        return copy.deepcopy(self.ports)

    # ----------------------------------------------------------- memory
    def mem_info(self) -> Dict[str, int]:
        return self.memory.copy()

    # ----------------------------------------------------------- cron
    def cron_list(self) -> List[Dict]:
        return copy.deepcopy(self.cron_entries)

    def cron_add(self, schedule: str, user: str, command: str) -> None:
        self.cron_entries.append({
            "schedule": schedule, "user": user, "command": command,
            "enabled": True, "last_run": None, "last_status": None,
        })

    def cron_remove(self, index: int) -> Tuple[bool, str]:
        if index < 0 or index >= len(self.cron_entries):
            return False, "Invalid cron entry index"
        self.cron_entries.pop(index)
        return True, ""

    # ----------------------------------------------------------- users
    def user_info(self, name: str) -> Tuple[bool, Dict]:
        if name not in self.users:
            return False, {}
        return True, self.users[name].copy()

    def current_user(self) -> str:
        return "user"

    # ----------------------------------------------------------- snapshot / reset
    def _deep_snapshot(self) -> Dict:
        return {
            "index": copy.deepcopy(self.index),
            "paths": copy.deepcopy(self.paths),
            "svc": copy.deepcopy(self.svc),
            "disk_mounts": copy.deepcopy(self.disk_mounts),
            "ports": copy.deepcopy(self.ports),
            "memory": copy.deepcopy(self.memory),
            "cron_entries": copy.deepcopy(self.cron_entries),
            "users": copy.deepcopy(self.users),
            "groups": copy.deepcopy(self.groups),
            "env_vars": copy.deepcopy(self.env_vars),
            "iptables_rules": copy.deepcopy(self.iptables_rules),
            "dns_records": copy.deepcopy(self.dns_records),
        }

    def snapshot(self) -> Dict:
        procs_lines = []
        for pname in sorted(self.svc):
            p = self.svc[pname]
            procs_lines.append(f"{p['pid']}\t{pname}\t\t{p['status'].upper()}")
        return {
            "files": {p: f["content"][:200] for p, f in self.index.items()},
            "directories": sorted(self.paths),
            "processes": "\n".join(procs_lines),
            "disk": self.disk_mounts,
            "memory": self.memory,
        }

    def clear(self):
        self.index.clear()
        self.paths.clear()
        self.svc.clear()
        self.disk_mounts.clear()
        self.ports.clear()
        self.memory.clear()
        self.cron_entries.clear()
        self.users.clear()
        self.groups.clear()
        self.env_vars.clear()
        self.iptables_rules.clear()
        self.dns_records.clear()
        self._setup()

    def hard_reset(self):
        """Reset to the exact initial state captured at construction."""
        if self._initial_state:
            for key, val in self._initial_state.items():
                setattr(self, key, copy.deepcopy(val))
