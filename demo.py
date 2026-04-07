from src.environment import TrainingEnv


def easy_demo():
    print("=" * 60)
    print("DEMO: Easy Task - Log Analysis")
    print("=" * 60)
    
    env = TrainingEnv(scenario="log_analysis")
    res = env.reset()
    
    print(f"\nTask: {res['info']['task_name']}")
    print(f"Instructions:\n{res['info']['instructions']}")
    
    cmds = [
        "cat /var/log/app.log",
        "grep 500 /var/log/app.log",
    ]
    
    for cmd in cmds:
        print(f"\n$ {cmd}")
        result = env.step(cmd)
        print(f"Reward: {result['reward']:.3f}")
        print(f"Task Score: {result['info']['task_score']}")
        print(f"Done: {result['done']}")
        
        if result['done']:
            print("\n✓ Task completed successfully!")
            break


def medium_demo():
    print("\n" + "=" * 60)
    print("DEMO: Medium Task - Permission Repair")
    print("=" * 60)
    
    env = TrainingEnv(scenario="permission_repair")
    res = env.reset()
    
    print(f"\nTask: {res['info']['task_name']}")
    print(f"Instructions:\n{res['info']['instructions']}")
    
    cmds = [
        "ls -la /home/user/scripts/",
        "chmod 0755 /home/user/scripts/cleanup.sh",
        "ls -la /home/user/scripts/cleanup.sh",
    ]
    
    for cmd in cmds:
        print(f"\n$ {cmd}")
        result = env.step(cmd)
        out = result['observation']
        print(f"Output:\n{out}")
        print(f"Task Score: {result['info']['task_score']}")
        print(f"Done: {result['done']}")
        
        if result['done']:
            print("\n✓ Task completed successfully!")
            break


def hard_demo():
    print("\n" + "=" * 60)
    print("DEMO: Hard Task - Process Recovery")
    print("=" * 60)
    
    env = TrainingEnv(scenario="process_recovery")
    res = env.reset()
    
    print(f"\nTask: {res['info']['task_name']}")
    print(f"Instructions:\n{res['info']['instructions']}")
    
    cmds = [
        "ps",
        "ps | grep postgres",
        "systemctl restart postgres",
        "ps | grep postgres",
        "systemctl status postgres",
    ]
    
    for cmd in cmds:
        print(f"\n$ {cmd}")
        result = env.step(cmd)
        print(f"Output:\n{result['observation']}")
        print(f"Task Score: {result['info']['task_score']}")
        print(f"Done: {result['done']}")
        
        if result['done']:
            print("\n✓ Task completed successfully!")
            break


def agent_demo():
    print("\n" + "=" * 60)
    print("DEMO: LLM Agent Interaction Pattern")
    print("=" * 60)
    
    env = TrainingEnv(scenario="permission_repair")
    res = env.reset()
    
    print("\nAgent workflow:")
    print("1. Reset environment")
    view = res["observation"]
    print(f"   Observation: {view}")
    
    print("\n2. Read task instructions from info")
    print(f"   Task: {res['info']['task_name']}")
    
    print("\n3. Plan: Figure out what commands to run")
    print("   -> Need to check permissions on /home/user/scripts/cleanup.sh")
    print("   -> Need to make it executable with chmod")
    
    print("\n4. Execute commands in sequence:")
    step = 0
    while step < 3:
        if step == 0:
            act = "ls -la /home/user/scripts/cleanup.sh"
        elif step == 1:
            act = "chmod 0755 /home/user/scripts/cleanup.sh"
        else:
            act = "ls -la /home/user/scripts/cleanup.sh"
        
        print(f"\n   Step {step + 1}: Execute '{act}'")
        result = env.step(act)
        
        print(f"   Reward: {result['reward']:.3f}")
        print(f"   Task Score: {result['info']['task_score']}")
        print(f"   Done: {result['done']}")
        
        step += 1
        
        if result['done']:
            print(f"\n✓ Agent completed task in {step} steps!")
            break


if __name__ == "__main__":
    print("\nLinux SRE Environment - Demo\n")
    
    easy_demo()
    medium_demo()
    hard_demo()
    
    agent_demo()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
