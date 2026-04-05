import re
from datetime import datetime
from typing import Tuple, List, Optional, Dict
from .virtual_filesystem import SystemStore


class Shell:

    def __init__(self, storage: SystemStore):
        self.store = storage
        self.cwd = "/home/user"
        self.log: List[str] = []
        self.ext = 0
        self._handlers = {
            "cat": self._cat, "ls": self._ls, "grep": self._grep,
            "chmod": self._chmod, "ps": self._ps, "kill": self._kill,
            "systemctl": self._systemctl, "pwd": self._pwd,
            "echo": self._echo, "test": self._test, "[": self._test,
            # new commands
            "cd": self._cd, "tail": self._tail, "head": self._head,
            "find": self._find, "wc": self._wc, "df": self._df,
            "du": self._du, "top": self._top, "free": self._free,
            "uptime": self._uptime, "whoami": self._whoami,
            "id": self._id, "hostname": self._hostname, "date": self._date,
            "touch": self._touch, "mkdir": self._mkdir, "cp": self._cp,
            "mv": self._mv, "rm": self._rm,
            "netstat": self._netstat, "ss": self._netstat,
            "curl": self._curl, "journalctl": self._journalctl,
            "crontab": self._crontab, "iptables": self._iptables,
            "mount": self._mount, "env": self._env, "export": self._export,
            "sort": self._sort, "uniq": self._uniq, "cut": self._cut,
            "tr": self._tr, "tee": self._tee, "which": self._which,
            "dig": self._dig, "nslookup": self._dig,
            "history": self._history_cmd, "clear": self._clear,
        }

    # ===================================================================
    #  CORE EXECUTION — pipes, redirects, chaining, var expansion
    # ===================================================================
    def run(self, text: str) -> Tuple[str, int]:
        self.log.append(text)
        text = text.strip()
        if not text:
            return "", 0

        # expand $VAR references
        text = self._expand_vars(text)

        # split on ; for command chaining
        segments = self._split_chain(text)
        final_out = ""
        final_code = 0
        for seg in segments:
            out, code = self._run_pipeline(seg.strip())
            final_out = out
            final_code = code
        return final_out, final_code

    def _run_pipeline(self, text: str) -> Tuple[str, int]:
        """Handle a single chained segment that may contain pipes and redirects."""
        if not text:
            return "", 0

        # extract redirect from the last pipe segment
        parts = self._split_pipes(text)
        redirect_file = None
        redirect_append = False
        last = parts[-1].strip()

        if ">>" in last:
            idx = last.index(">>")
            redirect_file = last[idx+2:].strip()
            parts[-1] = last[:idx].strip()
            redirect_append = True
        elif ">" in last:
            idx = last.index(">")
            redirect_file = last[idx+1:].strip()
            parts[-1] = last[:idx].strip()
            redirect_append = False

        pipe_input: Optional[str] = None
        code = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            tokens = self._tokenize(part)
            if not tokens:
                continue
            out, code = self._dispatch(tokens, pipe_input)
            pipe_input = out

        output = pipe_input if pipe_input is not None else ""

        if redirect_file:
            rpath = self._resolve(redirect_file)
            self.store.write_file(rpath, output + "\n", append=redirect_append)
            return "", code

        return output, code

    def _dispatch(self, tokens: List[str], pipe_input: Optional[str] = None) -> Tuple[str, int]:
        op = tokens[0]
        args = tokens[1:]
        handler = self._handlers.get(op)
        if handler is None:
            return f"bash: command not found: {op}", 127
        # pass pipe_input for commands that accept stdin
        if pipe_input is not None and op in ("grep", "sort", "uniq", "cut",
                                             "tr", "wc", "head", "tail", "tee"):
            return handler(args, pipe_input=pipe_input)
        return handler(args)

    # ===================================================================
    #  TOKENIZER  /  SPLITTERS
    # ===================================================================
    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        curr = ""
        quoted = False
        qchar: Optional[str] = None
        for char in text:
            if char in ('"', "'") and not quoted:
                quoted = True
                qchar = char
            elif char == qchar and quoted:
                quoted = False
                qchar = None
            elif char == " " and not quoted:
                if curr:
                    tokens.append(curr)
                    curr = ""
            else:
                curr += char
        if curr:
            tokens.append(curr)
        return tokens

    def _split_pipes(self, text: str) -> List[str]:
        """Split on | respecting quotes."""
        parts: List[str] = []
        curr = ""
        quoted = False
        qchar: Optional[str] = None
        for ch in text:
            if ch in ('"', "'") and not quoted:
                quoted, qchar = True, ch
                curr += ch
            elif ch == qchar and quoted:
                quoted, qchar = False, None
                curr += ch
            elif ch == "|" and not quoted:
                parts.append(curr)
                curr = ""
            else:
                curr += ch
        if curr:
            parts.append(curr)
        return parts

    def _split_chain(self, text: str) -> List[str]:
        """Split on ; and && respecting quotes."""
        parts: List[str] = []
        curr = ""
        quoted = False
        qchar: Optional[str] = None
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in ('"', "'") and not quoted:
                quoted, qchar = True, ch
                curr += ch
            elif ch == qchar and quoted:
                quoted, qchar = False, None
                curr += ch
            elif ch == ";" and not quoted:
                parts.append(curr)
                curr = ""
            elif ch == "&" and i+1 < len(text) and text[i+1] == "&" and not quoted:
                parts.append(curr)
                curr = ""
                i += 1  # skip second &
            else:
                curr += ch
            i += 1
        if curr:
            parts.append(curr)
        return parts

    def _expand_vars(self, text: str) -> str:
        def _replace(m):
            var = m.group(1)
            return self.store.env_vars.get(var, "")
        return re.sub(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?', _replace, text)

    def _resolve(self, path: str) -> str:
        if path.startswith("/"):
            resolved = path
        elif path == ".":
            resolved = self.cwd
        elif path == "..":
            parts = self.cwd.strip("/").split("/")
            resolved = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        elif path.startswith("~/"):
            resolved = "/home/user" + path[1:]
        else:
            resolved = self.cwd.rstrip("/") + "/" + path
        # normalise (remove double slashes, trailing slash)
        parts = [p for p in resolved.split("/") if p and p != "."]
        normalised: List[str] = []
        for p in parts:
            if p == "..":
                if normalised:
                    normalised.pop()
            else:
                normalised.append(p)
        return "/" + "/".join(normalised) if normalised else "/"

    # ===================================================================
    #  ORIGINAL COMMANDS  (cat, ls, grep, chmod, ps, kill, systemctl, …)
    # ===================================================================
    def _cat(self, args: List[str], **kw) -> Tuple[str, int]:
        if not args:
            return "Usage: cat <file>", 1
        result = []
        for arg in args:
            if arg.startswith("-"):
                continue
            path = self._resolve(arg)
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            result.append(content.rstrip())
        return "\n".join(result), 0

    def _ls(self, args: List[str], **kw) -> Tuple[str, int]:
        show_long = False
        show_all = False
        targets = []
        for a in args:
            if a.startswith("-"):
                if "l" in a:
                    show_long = True
                if "a" in a:
                    show_all = True
            else:
                targets.append(a)
        if not targets:
            targets = ["."]

        all_output: List[str] = []
        for t in targets:
            path = self._resolve(t)
            # is it a file?
            if self.store.exists(path):
                if show_long:
                    all_output.append(self._ls_long_line(path))
                else:
                    all_output.append(path.split("/")[-1])
                continue
            ok, items = self.store.entries(path)
            if not ok:
                return items, 1
            if not show_all:
                items = [i for i in items if not i.startswith(".")]
            if show_long:
                for item in items:
                    full = path.rstrip("/") + "/" + item.rstrip("/")
                    all_output.append(self._ls_long_line(
                        full, is_dir=item.endswith("/")))
            else:
                all_output.extend(items)
        return "\n".join(all_output), 0

    def _ls_long_line(self, path: str, is_dir: bool = False) -> str:
        ok, info = self.store.info(path)
        if not ok:
            name = path.split("/")[-1].rstrip("/")
            return f"drwxr-xr-x 2 user user 4096 Mar 30 08:00 {name}/" if is_dir else name
        perms = info.get("permissions", 0o644)
        perm_str = self._perm_string(perms, is_dir)
        owner = info.get("owner", "user")
        group = info.get("group", "user")
        size = info.get("size", len(info.get("content", "")))
        name = path.split("/")[-1]
        return f"{perm_str} 1 {owner} {group} {size:>8} Mar 30 08:00 {name}"

    @staticmethod
    def _perm_string(mode: int, is_dir: bool = False) -> str:
        chars = "d" if is_dir else "-"
        for shift in (6, 3, 0):
            triplet = (mode >> shift) & 0o7
            chars += "r" if triplet & 4 else "-"
            chars += "w" if triplet & 2 else "-"
            chars += "x" if triplet & 1 else "-"
        return chars

    def _grep(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        # parse flags
        invert = False
        count_only = False
        ignore_case = False
        line_numbers = False
        pattern = None
        file_arg = None
        i = 0
        while i < len(args):
            a = args[i]
            if a == "-v":
                invert = True
            elif a == "-c":
                count_only = True
            elif a == "-i":
                ignore_case = True
            elif a == "-n":
                line_numbers = True
            elif a == "-E" or a == "-P":
                pass  # extended/perl regex, just accept
            elif pattern is None:
                pattern = a
            else:
                file_arg = a
            i += 1

        if pattern is None:
            return "Usage: grep <pattern> <file>", 1

        if pipe_input is not None:
            lines = pipe_input.split("\n")
        elif file_arg:
            path = self._resolve(file_arg)
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            lines = content.split("\n")
        else:
            return "Usage: grep <pattern> <file>", 1

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"grep: invalid regex: {e}", 1

        matched = []
        for idx, line in enumerate(lines, 1):
            hit = bool(regex.search(line))
            if invert:
                hit = not hit
            if hit:
                if line_numbers:
                    matched.append(f"{idx}:{line}")
                else:
                    matched.append(line)

        if count_only:
            return str(len(matched)), 0

        if not matched:
            return "", 1
        return "\n".join(matched), 0

    def _chmod(self, args: List[str], **kw) -> Tuple[str, int]:
        recursive = False
        mode_txt = None
        targets = []
        for a in args:
            if a == "-R":
                recursive = True
            elif mode_txt is None:
                mode_txt = a
            else:
                targets.append(a)
        if not mode_txt or not targets:
            return "Usage: chmod <mode> <file>", 1
        try:
            if mode_txt.startswith("0o") or mode_txt.startswith("0"):
                mode = int(mode_txt, 8)
            elif mode_txt.startswith("+"):
                # handle +x
                if "x" in mode_txt:
                    path0 = self._resolve(targets[0])
                    ok, curr = self.store.perms(path0)
                    mode = (curr | 0o111) if ok else 0o755
                else:
                    return f"chmod: invalid mode '{mode_txt}'", 1
            else:
                mode = int(mode_txt, 8)
        except ValueError:
            return f"chmod: invalid mode '{mode_txt}'", 1
        for t in targets:
            path = self._resolve(t)
            ok, err = self.store.chmod(path, mode)
            if not ok:
                return err, 1
        return "", 0

    def _ps(self, args: List[str], **kw) -> Tuple[str, int]:
        show_all = any(a in ("aux", "-aux", "-ef", "-e", "aux") for a in args)
        lines = ["PID\tNAME\t\tSTATUS\tCPU%\tMEM(MB)\tUSER"]
        for pname in sorted(self.store.svc_list()):
            ok, pinfo = self.store.svc_info(pname)
            if ok:
                pid = pinfo.get("pid", "?")
                stat = pinfo.get("status", "?").upper()
                cpu = pinfo.get("cpu", 0.0)
                mem = pinfo.get("mem_mb", 0)
                usr = pinfo.get("user", "?")
                lines.append(f"{pid}\t{pname}\t\t{stat}\t{cpu}\t{mem}\t{usr}")
        return "\n".join(lines), 0

    def _kill(self, args: List[str], **kw) -> Tuple[str, int]:
        if not args:
            return "Usage: kill <process_name|pid>", 1
        signal = "SIGTERM"
        target = args[0]
        if target.startswith("-"):
            signal = target
            target = args[1] if len(args) > 1 else ""
        if not target:
            return "Usage: kill <process_name|pid>", 1
        # try by name first
        ok, err = self.store.svc_stop(target)
        if not ok:
            # try by pid
            for sname in self.store.svc_list():
                _, info = self.store.svc_info(sname)
                if info and str(info.get("pid")) == target:
                    ok, err = self.store.svc_stop(sname)
                    break
        if not ok:
            return err or f"kill: ({target}) - No such process", 1
        return "", 0

    def _systemctl(self, args: List[str], **kw) -> Tuple[str, int]:
        if len(args) < 2:
            return "Usage: systemctl <start|stop|restart|status|enable|disable> <service>", 1
        act = args[0]
        svc = args[1]
        if act == "start":
            ok, err = self.store.svc_start(svc)
            return (f"Started {svc}.", 0) if ok else (err, 1)
        elif act == "stop":
            ok, err = self.store.svc_stop(svc)
            return (f"Stopped {svc}.", 0) if ok else (err, 1)
        elif act == "restart":
            self.store.svc_stop(svc)
            ok, err = self.store.svc_start(svc)
            return (f"Restarted {svc}.", 0) if ok else (err, 1)
        elif act == "status":
            ok, pinfo = self.store.svc_info(svc)
            if not ok:
                return f"Unit {svc}.service could not be found.", 1
            stat = pinfo.get("status", "unknown")
            pid = pinfo.get("pid", "?")
            mem = pinfo.get("mem_mb", 0)
            active = "active (running)" if stat == "running" else "inactive (dead)"
            return (
                f"● {svc}.service\n"
                f"   Loaded: loaded (/etc/systemd/system/{svc}.service; enabled)\n"
                f"   Active: {active}\n"
                f" Main PID: {pid}\n"
                f"   Memory: {mem}M\n"
            ), 0
        elif act in ("enable", "disable"):
            ok, _ = self.store.svc_info(svc)
            if not ok:
                return f"Unit {svc}.service could not be found.", 1
            return f"{'Enabled' if act == 'enable' else 'Disabled'} {svc}.service.", 0
        elif act == "list-units":
            return self._ps(["aux"])
        else:
            return f"Unknown action: {act}", 1

    def _pwd(self, args: List[str], **kw) -> Tuple[str, int]:
        return self.cwd, 0

    def _echo(self, args: List[str], **kw) -> Tuple[str, int]:
        return " ".join(args), 0

    def _test(self, args: List[str], **kw) -> Tuple[str, int]:
        clean = [a for a in args if a != "]"]
        if len(clean) < 2:
            return "", 1
        flag = clean[0]
        path = self._resolve(clean[1])
        if flag == "-f":
            return "", 0 if self.store.exists(path) else 1
        elif flag == "-x":
            ok, perms = self.store.perms(path)
            return "", 0 if (ok and perms & 0o111) else 1
        elif flag == "-d":
            return "", 0 if self.store.dir_exists(path) else 1
        elif flag == "-e":
            return "", 0 if (self.store.exists(path) or self.store.dir_exists(path)) else 1
        elif flag == "-s":
            ok, info = self.store.info(path)
            return "", 0 if (ok and len(info.get("content", "")) > 0) else 1
        elif flag == "-r":
            return "", 0 if self.store.exists(path) else 1
        elif flag == "-w":
            return "", 0 if self.store.exists(path) else 1
        else:
            return f"test: unknown operator '{flag}'", 1

    # ===================================================================
    #  NEW COMMANDS
    # ===================================================================
    def _cd(self, args: List[str], **kw) -> Tuple[str, int]:
        if not args or args[0] == "~":
            self.cwd = "/home/user"
            return "", 0
        target = self._resolve(args[0])
        if not self.store.dir_exists(target):
            return f"bash: cd: {args[0]}: No such file or directory", 1
        self.cwd = target
        return "", 0

    def _tail(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        n = 10
        file_arg = None
        i = 0
        while i < len(args):
            if args[i] == "-n" and i+1 < len(args):
                try:
                    n = int(args[i+1])
                except ValueError:
                    pass
                i += 2
                continue
            elif args[i] == "-f":
                i += 1
                continue  # simulate: just show last lines
            elif not args[i].startswith("-"):
                file_arg = args[i]
            i += 1
        if pipe_input is not None:
            lines = pipe_input.split("\n")
        elif file_arg:
            path = self._resolve(file_arg)
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            lines = content.split("\n")
        else:
            return "Usage: tail [-n N] <file>", 1
        result = lines[-n:] if len(lines) > n else lines
        return "\n".join(result), 0

    def _head(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        n = 10
        file_arg = None
        i = 0
        while i < len(args):
            if args[i] == "-n" and i+1 < len(args):
                try:
                    n = int(args[i+1])
                except ValueError:
                    pass
                i += 2
                continue
            elif not args[i].startswith("-"):
                file_arg = args[i]
            i += 1
        if pipe_input is not None:
            lines = pipe_input.split("\n")
        elif file_arg:
            path = self._resolve(file_arg)
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            lines = content.split("\n")
        else:
            return "Usage: head [-n N] <file>", 1
        result = lines[:n]
        return "\n".join(result), 0

    def _find(self, args: List[str], **kw) -> Tuple[str, int]:
        search_dir = "."
        name_pattern = None
        file_type = None
        i = 0
        while i < len(args):
            if args[i] == "-name" and i+1 < len(args):
                name_pattern = args[i+1]
                i += 2
            elif args[i] == "-type" and i+1 < len(args):
                file_type = args[i+1]
                i += 2
            elif not args[i].startswith("-"):
                search_dir = args[i]
                i += 1
            else:
                i += 1
        base = self._resolve(search_dir)
        results: List[str] = []
        if file_type != "f":
            for d in sorted(self.store.paths):
                if d.startswith(base) or d == base:
                    if name_pattern:
                        dname = d.split("/")[-1] if d != "/" else "/"
                        if not self._glob_match(name_pattern, dname):
                            continue
                    results.append(d)
        if file_type != "d":
            for fpath in sorted(self.store.index.keys()):
                if fpath.startswith(base):
                    if name_pattern:
                        fname = fpath.split("/")[-1]
                        if not self._glob_match(name_pattern, fname):
                            continue
                    results.append(fpath)
        if not results:
            return "", 0
        return "\n".join(results), 0

    @staticmethod
    def _glob_match(pattern: str, name: str) -> bool:
        regex = pattern.replace(".", r"\.").replace(
            "*", ".*").replace("?", ".")
        return bool(re.match(f"^{regex}$", name))

    def _wc(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        show_l = "-l" in args
        show_w = "-w" in args
        show_c = "-c" in args
        if not (show_l or show_w or show_c):
            show_l = show_w = show_c = True
        file_args = [a for a in args if not a.startswith("-")]
        if pipe_input is not None:
            content = pipe_input
            label = ""
        elif file_args:
            path = self._resolve(file_args[0])
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            label = f" {file_args[0]}"
        else:
            return "Usage: wc [-lwc] <file>", 1
        lines_count = content.count("\n")
        words_count = len(content.split())
        chars_count = len(content)
        parts = []
        if show_l:
            parts.append(f"{lines_count:>8}")
        if show_w:
            parts.append(f"{words_count:>8}")
        if show_c:
            parts.append(f"{chars_count:>8}")
        return "".join(parts) + label, 0

    def _df(self, args: List[str], **kw) -> Tuple[str, int]:
        human = "-h" in args
        lines = ["Filesystem      Size  Used Avail Use% Mounted on"]
        for mount, d in sorted(self.store.disk_mounts.items()):
            total = d["total_mb"]
            used = d["used_mb"]
            avail = d["available_mb"]
            pct = int((used / total) * 100) if total else 0
            if human:
                lines.append(
                    f"{d['device']:<16}{self._human_size(total):<6}"
                    f"{self._human_size(used):<6}{self._human_size(avail):<6}"
                    f"{pct}%   {mount}"
                )
            else:
                lines.append(
                    f"{d['device']:<16}{total*1024:<10}{used*1024:<10}"
                    f"{avail*1024:<10}{pct}%   {mount}"
                )
        return "\n".join(lines), 0

    def _du(self, args: List[str], **kw) -> Tuple[str, int]:
        human = "-h" in args
        summary = "-s" in args
        targets = [a for a in args if not a.startswith("-")]
        if not targets:
            targets = ["."]
        lines = []
        for t in targets:
            path = self._resolve(t)
            total = 0
            for fpath, finfo in self.store.index.items():
                if fpath.startswith(path):
                    total += len(finfo.get("content", ""))
            size_str = self._human_size(total // 1024) if human else str(total)
            lines.append(f"{size_str}\t{t}")
        return "\n".join(lines), 0

    def _top(self, args: List[str], **kw) -> Tuple[str, int]:
        mem = self.store.mem_info()
        load = self.store.load_avg
        lines = [
            f"top - {datetime.now().strftime('%H:%M:%S')} up 1 day,  load average: {load[0]}, {load[1]}, {load[2]}",
            f"Tasks: {len(self.store.svc)} total",
            f"Mem:  {mem['total_mb']}M total, {mem['free_mb']}M free, {mem['used_mb']}M used, {mem['cached_mb']}M cache",
            f"Swap: {mem['swap_total_mb']}M total, {mem['swap_free_mb']}M free, {mem['swap_used_mb']}M used",
            "",
            f"{'PID':>6} {'USER':<10} {'CPU%':>5} {'MEM':>6} {'STATUS':<10} COMMAND",
        ]
        for pname in sorted(self.store.svc_list()):
            ok, p = self.store.svc_info(pname)
            if ok:
                lines.append(
                    f"{p['pid']:>6} {p.get('user', '?'):<10} {p.get('cpu', 0.0):>5.1f} "
                    f"{p.get('mem_mb', 0):>5}M {p['status']:<10} {pname}"
                )
        return "\n".join(lines), 0

    def _free(self, args: List[str], **kw) -> Tuple[str, int]:
        mem = self.store.mem_info()
        human = "-h" in args or "-m" in args
        lines = [
            f"{'':>15}{'total':>12}{'used':>12}{'free':>12}{'available':>12}",
            f"{'Mem:':>15}{mem['total_mb']:>12}{mem['used_mb']:>12}"
            f"{mem['free_mb']:>12}{mem['available_mb']:>12}",
            f"{'Swap:':>15}{mem['swap_total_mb']:>12}{mem['swap_used_mb']:>12}"
            f"{mem['swap_free_mb']:>12}{'':>12}",
        ]
        return "\n".join(lines), 0

    def _uptime(self, args: List[str], **kw) -> Tuple[str, int]:
        secs = self.store.uptime_secs
        days = secs // 86400
        hours = (secs % 86400) // 3600
        mins = (secs % 3600) // 60
        load = self.store.load_avg
        running = sum(1 for s in self.store.svc.values()
                      if s["status"] == "running")
        return (
            f" {datetime.now().strftime('%H:%M:%S')} up {days} day(s), {hours}:{mins:02d}, "
            f"{running} users,  load average: {load[0]}, {load[1]}, {load[2]}"
        ), 0

    def _whoami(self, args: List[str], **kw) -> Tuple[str, int]:
        return self.store.current_user(), 0

    def _id(self, args: List[str], **kw) -> Tuple[str, int]:
        u = args[0] if args else self.store.current_user()
        ok, info = self.store.user_info(u)
        if not ok:
            return f"id: '{u}': no such user", 1
        grps = [g for g, members in self.store.groups.items() if u in members]
        return f"uid={info['uid']}({u}) gid={info['gid']}({u}) groups={','.join(grps)}", 0

    def _hostname(self, args: List[str], **kw) -> Tuple[str, int]:
        return self.store.hostname, 0

    def _date(self, args: List[str], **kw) -> Tuple[str, int]:
        fmt = None
        for a in args:
            if a.startswith("+"):
                fmt = a[1:]
        if fmt:
            return datetime.now().strftime(fmt), 0
        return datetime.now().strftime("%a %b %d %H:%M:%S UTC %Y"), 0

    def _touch(self, args: List[str], **kw) -> Tuple[str, int]:
        if not args:
            return "Usage: touch <file>", 1
        for a in args:
            if a.startswith("-"):
                continue
            path = self._resolve(a)
            ok, err = self.store.touch(path)
            if not ok:
                return err, 1
        return "", 0

    def _mkdir(self, args: List[str], **kw) -> Tuple[str, int]:
        make_parents = "-p" in args
        targets = [a for a in args if not a.startswith("-")]
        if not targets:
            return "Usage: mkdir [-p] <directory>", 1
        for t in targets:
            path = self._resolve(t)
            if make_parents:
                ok, err = self.store.mkdir_p(path)
            else:
                ok, err = self.store.mkdir(path)
            if not ok:
                return err, 1
        return "", 0

    def _cp(self, args: List[str], **kw) -> Tuple[str, int]:
        files = [a for a in args if not a.startswith("-")]
        if len(files) < 2:
            return "Usage: cp <source> <dest>", 1
        src = self._resolve(files[0])
        dst = self._resolve(files[1])
        ok, err = self.store.cp(src, dst)
        return ("", 0) if ok else (err, 1)

    def _mv(self, args: List[str], **kw) -> Tuple[str, int]:
        files = [a for a in args if not a.startswith("-")]
        if len(files) < 2:
            return "Usage: mv <source> <dest>", 1
        src = self._resolve(files[0])
        dst = self._resolve(files[1])
        ok, err = self.store.mv(src, dst)
        return ("", 0) if ok else (err, 1)

    def _rm(self, args: List[str], **kw) -> Tuple[str, int]:
        targets = [a for a in args if not a.startswith("-")]
        if not targets:
            return "Usage: rm <file>", 1
        for t in targets:
            path = self._resolve(t)
            ok, err = self.store.rm(path)
            if not ok:
                return err, 1
        return "", 0

    def _netstat(self, args: List[str], **kw) -> Tuple[str, int]:
        lines = ["Proto  Local Address          State       PID/Program"]
        for port, info in sorted(self.store.all_ports().items()):
            addr = f"{info['addr']}:{port}"
            svc = info["service"]
            state = info["state"]
            proto = info["proto"]
            lines.append(f"{proto:<7}{addr:<23}{state:<12}{svc}")
        return "\n".join(lines), 0

    def _curl(self, args: List[str], **kw) -> Tuple[str, int]:
        silent = "-s" in args or "-sf" in args
        fail_silent = "-sf" in args or "-f" in args
        url = None
        for a in args:
            if not a.startswith("-"):
                url = a
                break
        if not url:
            return "Usage: curl [-sf] <url>", 1
        # simulate responses for known local services
        if "localhost:8080" in url or "127.0.0.1:8080" in url:
            ok, info = self.store.svc_info("app")
            if ok and info["status"] == "running":
                if "/health" in url:
                    return '{"status":"ok"}', 0
                return '{"message":"Application is running"}', 0
            else:
                if fail_silent:
                    return "", 7
                return "curl: (7) Failed to connect to localhost port 8080: Connection refused", 7
        elif "localhost:80" in url or "127.0.0.1:80" in url or "localhost" == url:
            ok, info = self.store.svc_info("nginx")
            if ok and info["status"] == "running":
                return "<html><body>Welcome to nginx</body></html>", 0
            return "curl: (7) Failed to connect to localhost port 80: Connection refused", 7
        elif "localhost:5432" in url:
            return "curl: (52) Empty reply from server", 52
        elif "localhost:6379" in url:
            ok, info = self.store.svc_info("redis")
            if ok and info["status"] == "running":
                return "+PONG", 0
            return "curl: (7) Failed to connect to localhost port 6379: Connection refused", 7
        else:
            return f"curl: (6) Could not resolve host: {url}", 6

    def _journalctl(self, args: List[str], **kw) -> Tuple[str, int]:
        unit = None
        n_lines = 20
        since = None
        i = 0
        while i < len(args):
            if args[i] == "-u" and i+1 < len(args):
                unit = args[i+1]
                i += 2
            elif args[i] == "-n" and i+1 < len(args):
                try:
                    n_lines = int(args[i+1])
                except ValueError:
                    pass
                i += 2
            elif args[i] == "--since" and i+1 < len(args):
                since = args[i+1]
                i += 2
            elif args[i] in ("-xe", "-x", "-e", "--no-pager"):
                i += 1
            else:
                i += 1
        # pull from syslog
        ok, content = self.store.fetch("/var/log/syslog")
        if not ok:
            return "-- No entries --", 0
        lines = content.strip().split("\n")
        if unit:
            lines = [l for l in lines if unit in l]
        lines = lines[-n_lines:]
        if not lines:
            return f"-- No entries for {unit or 'journal'} --", 0
        return "\n".join(lines), 0

    def _crontab(self, args: List[str], **kw) -> Tuple[str, int]:
        if "-l" in args:
            entries = self.store.cron_list()
            if not entries:
                return "no crontab for user", 0
            lines = ["# Crontab entries:"]
            for e in entries:
                status = "# DISABLED " if not e["enabled"] else ""
                last = f"  # last: {e['last_status']}" if e["last_status"] else ""
                lines.append(f"{status}{e['schedule']} {e['command']}{last}")
            return "\n".join(lines), 0
        elif "-e" in args:
            return "crontab: editor mode not available in this environment", 1
        elif "-r" in args:
            return "crontab: removed", 0
        ok, content = self.store.fetch("/etc/crontab")
        if ok:
            return content, 0
        return "no crontab", 0

    def _iptables(self, args: List[str], **kw) -> Tuple[str, int]:
        if "-L" in args or "--list" in args or not args:
            return "\n".join(self.store.iptables_rules), 0
        return "iptables: Operation not permitted (simulated read-only)", 1

    def _mount(self, args: List[str], **kw) -> Tuple[str, int]:
        lines = []
        for mnt, d in sorted(self.store.disk_mounts.items()):
            lines.append(
                f"{d['device']} on {mnt} type {d['fstype']} (rw,relatime)")
        return "\n".join(lines), 0

    def _env(self, args: List[str], **kw) -> Tuple[str, int]:
        lines = [f"{k}={v}" for k, v in sorted(self.store.env_vars.items())]
        return "\n".join(lines), 0

    def _export(self, args: List[str], **kw) -> Tuple[str, int]:
        for a in args:
            if "=" in a:
                key, val = a.split("=", 1)
                self.store.env_vars[key] = val
            else:
                return f"export: '{a}': not a valid identifier", 1
        return "", 0

    def _sort(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        reverse = "-r" in args
        numeric = "-n" in args
        file_args = [a for a in args if not a.startswith("-")]
        if pipe_input is not None:
            lines = pipe_input.split("\n")
        elif file_args:
            path = self._resolve(file_args[0])
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            lines = content.split("\n")
        else:
            return "", 0
        if numeric:
            def sort_key(l):
                try:
                    return float(l.split()[0])
                except (ValueError, IndexError):
                    return 0
            lines.sort(key=sort_key, reverse=reverse)
        else:
            lines.sort(reverse=reverse)
        return "\n".join(lines), 0

    def _uniq(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        count = "-c" in args
        file_args = [a for a in args if not a.startswith("-")]
        if pipe_input is not None:
            lines = pipe_input.split("\n")
        elif file_args:
            path = self._resolve(file_args[0])
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            lines = content.split("\n")
        else:
            return "", 0
        result = []
        prev = None
        cnt = 0
        for line in lines:
            if line == prev:
                cnt += 1
            else:
                if prev is not None:
                    result.append(f"{cnt:>7} {prev}" if count else prev)
                prev = line
                cnt = 1
        if prev is not None:
            result.append(f"{cnt:>7} {prev}" if count else prev)
        return "\n".join(result), 0

    def _cut(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        delim = "\t"
        fields = None
        file_arg = None
        i = 0
        while i < len(args):
            if args[i] == "-d" and i+1 < len(args):
                delim = args[i+1]
                i += 2
            elif args[i] == "-f" and i+1 < len(args):
                fields = args[i+1]
                i += 2
            elif not args[i].startswith("-"):
                file_arg = args[i]
                i += 1
            else:
                i += 1

        if pipe_input is not None:
            lines = pipe_input.split("\n")
        elif file_arg:
            path = self._resolve(file_arg)
            ok, content = self.store.fetch(path)
            if not ok:
                return content, 1
            lines = content.split("\n")
        else:
            return "", 0

        if not fields:
            return "\n".join(lines), 0

        # parse field list (1-indexed)
        field_indices = set()
        for part in fields.split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                s = int(start) if start else 1
                e = int(end) if end else 100
                field_indices.update(range(s, e+1))
            else:
                field_indices.add(int(part))

        result = []
        for line in lines:
            parts = line.split(delim)
            selected = [parts[i-1]
                        for i in sorted(field_indices) if i-1 < len(parts)]
            result.append(delim.join(selected))
        return "\n".join(result), 0

    def _tr(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        delete = "-d" in args
        clean_args = [a for a in args if not a.startswith("-")]
        text = pipe_input or ""
        if delete and clean_args:
            for ch in clean_args[0]:
                text = text.replace(ch, "")
            return text, 0
        if len(clean_args) >= 2:
            src, dst = clean_args[0], clean_args[1]
            table = str.maketrans(src, dst[:len(src)])
            return text.translate(table), 0
        return text, 0

    def _tee(self, args: List[str], pipe_input: Optional[str] = None, **kw) -> Tuple[str, int]:
        append = "-a" in args
        targets = [a for a in args if not a.startswith("-")]
        text = pipe_input or ""
        for t in targets:
            path = self._resolve(t)
            self.store.write_file(path, text + "\n", append=append)
        return text, 0

    def _which(self, args: List[str], **kw) -> Tuple[str, int]:
        known = {
            "cat": "/usr/bin/cat", "grep": "/usr/bin/grep", "ls": "/usr/bin/ls",
            "chmod": "/usr/bin/chmod", "ps": "/usr/bin/ps", "kill": "/usr/bin/kill",
            "systemctl": "/usr/bin/systemctl", "curl": "/usr/bin/curl",
            "find": "/usr/bin/find", "python3": "/usr/bin/python3",
            "bash": "/usr/bin/bash", "ssh": "/usr/bin/ssh", "tar": "/usr/bin/tar",
            "gzip": "/usr/bin/gzip", "pg_dump": "/usr/bin/pg_dump",
        }
        if not args:
            return "Usage: which <command>", 1
        cmd = args[0]
        if cmd in known:
            return known[cmd], 0
        return f"which: no {cmd} in PATH", 1

    def _dig(self, args: List[str], **kw) -> Tuple[str, int]:
        host = None
        for a in args:
            if not a.startswith("-") and not a.startswith("+"):
                host = a
                break
        if not host:
            return "Usage: dig <hostname>", 1
        ip = self.store.dns_records.get(host)
        if ip:
            return (
                f";; ANSWER SECTION:\n{host}.\t\t300\tIN\tA\t{ip}\n\n"
                f";; Query time: 2 msec\n;; SERVER: 8.8.8.8#53"
            ), 0
        return f";; connection timed out; no servers could be reached for {host}", 1

    def _history_cmd(self, args: List[str], **kw) -> Tuple[str, int]:
        lines = []
        for i, cmd in enumerate(self.log, 1):
            lines.append(f"  {i}  {cmd}")
        return "\n".join(lines), 0

    def _clear(self, args: List[str], **kw) -> Tuple[str, int]:
        return "", 0

    # ===================================================================
    #  HELPERS
    # ===================================================================
    @staticmethod
    def _human_size(mb: int) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f}G"
        return f"{mb}M"

    def history(self) -> List[str]:
        return self.log.copy()
