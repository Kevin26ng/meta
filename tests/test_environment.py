import pytest
from src.virtual_filesystem import SystemStore
from src.terminal_emulator import Shell
from src.environment import TrainingEnv


class TestSystemStore:
    
    def test_read_file(self):
        store = SystemStore()
        ok, content = store.fetch("/var/log/app.log")
        assert ok
        assert "ERROR" in content
        assert "500" in content
    
    def test_file_not_found(self):
        store = SystemStore()
        ok, err = store.fetch("/nonexistent/file.txt")
        assert not ok
        assert "No such file" in err
    
    def test_permissions(self):
        store = SystemStore()
        
        ok, perms = store.perms("/home/user/scripts/cleanup.sh")
        assert ok
        assert perms == 0o644
        
        ok, err = store.chmod("/home/user/scripts/cleanup.sh", 0o755)
        assert ok
        
        ok, perms2 = store.perms("/home/user/scripts/cleanup.sh")
        assert perms2 == 0o755
    
    def test_list_directory(self):
        store = SystemStore()
        ok, items = store.entries("/home/user/scripts")
        assert ok
        assert "cleanup.sh" in items
    
    def test_process_status(self):
        store = SystemStore()
        ok, proc = store.svc_info("postgres")
        assert ok
        assert proc["status"] == "dead"
    
    def test_kill_process(self):
        store = SystemStore()
        ok, _ = store.svc_stop("nginx")
        assert ok
        
        ok, proc = store.svc_info("nginx")
        assert proc["status"] == "dead"
    
    def test_restart_process(self):
        store = SystemStore()
        
        store.svc_stop("nginx")
        ok, _ = store.svc_start("nginx")
        assert ok
        
        ok, proc = store.svc_info("nginx")
        assert proc["status"] == "running"


class TestShell:
    
    def test_cat_command(self):
        store = SystemStore()
        shell = Shell(store)
        
        out, code = shell.run("cat /var/log/app.log")
        assert code == 0
        assert "ERROR" in out
    
    def test_grep_command(self):
        store = SystemStore()
        shell = Shell(store)
        
        out, code = shell.run("grep 500 /var/log/app.log")
        assert code == 0
        assert "2026-03-30T09:22:19.234Z" in out
    
    def test_chmod_command(self):
        store = SystemStore()
        shell = Shell(store)
        
        out, code = shell.run("chmod 0755 /home/user/scripts/cleanup.sh")
        assert code == 0
        
        ok, perms = store.perms("/home/user/scripts/cleanup.sh")
        assert perms == 0o755
    
    def test_ps_command(self):
        store = SystemStore()
        shell = Shell(store)
        
        out, code = shell.run("ps")
        assert code == 0
        assert "nginx" in out
        assert "postgres" in out
        assert "redis" in out
    
    def test_systemctl_command(self):
        store = SystemStore()
        shell = Shell(store)
        
        out, code = shell.run("systemctl start postgres")
        assert code == 0
        assert "Started postgres" in out


class TestTrainingEnv:
    
    def test_reset(self):
        env = TrainingEnv(scenario="log_analysis")
        res = env.reset()
        
        assert "observation" in res
        assert "info" in res
        assert res["info"]["task_name"] == "Log Analysis"
    
    def test_step(self):
        env = TrainingEnv(scenario="log_analysis")
        env.reset()
        
        res = env.step("grep 500 /var/log/app.log")
        
        assert "observation" in res
        assert "reward" in res
        assert "done" in res
        assert "info" in res
    
    def test_log_analysis_completion(self):
        env = TrainingEnv(scenario="log_analysis")
        env.reset()
        
        res = env.step("grep 500 /var/log/app.log")
        
        assert res["info"]["task_score"] == 1.0
        assert res["done"]
    
    def test_permission_repair_completion(self):
        env = TrainingEnv(scenario="permission_repair")
        env.reset()
        
        res = env.step("chmod 0755 /home/user/scripts/cleanup.sh")
        
        assert res["info"]["task_score"] == 1.0
        assert res["done"]
    
    def test_process_recovery_stages(self):
        env = TrainingEnv(scenario="process_recovery")
        env.reset()
        
        res = env.step("ps | grep postgres")
        s1 = res["info"]["task_score"]
        assert s1 > 0
        
        res = env.step("systemctl restart postgres")
        s2 = res["info"]["task_score"]
        assert s2 >= s1
    
    def test_episode_termination(self):
        env = TrainingEnv(scenario="log_analysis")
        env.reset()
        
        res = env.step("grep 500 /var/log/app.log")
        assert res["done"]
        
        res = env.step("echo 'Test'")
        assert res["done"]



class TestIntegration:
    
    def test_full_log_analysis_workflow(self):
        env = TrainingEnv(scenario="log_analysis")
        env.reset()
        
        res = env.step("grep 500 /var/log/app.log")
        
        assert res["done"]
        assert res["info"]["task_score"] == 1.0
    
    def test_full_perm_repair_workflow(self):
        env = TrainingEnv(scenario="permission_repair")
        env.reset()
        
        env.step("ls -la /home/user/scripts/cleanup.sh")
        res = env.step("chmod 0755 /home/user/scripts/cleanup.sh")
        
        assert res["done"]
        assert res["info"]["task_score"] == 1.0
    
    def test_multiple_environments(self):
        e1 = TrainingEnv(scenario="log_analysis")
        e2 = TrainingEnv(scenario="permission_repair")
        
        r1 = e1.reset()
        r2 = e2.reset()
        
        assert r1["info"]["task_name"] == "Log Analysis"
        assert r2["info"]["task_name"] == "Permission Repair"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
