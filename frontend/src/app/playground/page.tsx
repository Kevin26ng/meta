"use client";

import { Activity, CircleCheck, Circle, Search, History, CheckCircle2, Brain, ChevronDown } from "lucide-react";
import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";

type LogEntry = {
  type: 'input' | 'output' | 'error' | 'system';
  text: string;
};

export default function PlaygroundPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-full w-full">
        <div className="text-chaos-green animate-pulse font-mono text-lg">Initializing Sandbox...</div>
      </div>
    }>
      <PlaygroundContent />
    </Suspense>
  );
}

function PlaygroundContent() {
  const searchParams = useSearchParams();
  const scenarioKey = searchParams.get('scenario') || 'log_analysis';
  const autoAgent = searchParams.get('auto_agent');

  const [envId, setEnvId] = useState<string | null>(null);
  const [scenarioMeta, setScenarioMeta] = useState<any>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [command, setCommand] = useState("");
  const [score, setScore] = useState(0);
  const [step, setStep] = useState(0);
  const [maxSteps, setMaxSteps] = useState(50);
  const [isReady, setIsReady] = useState(false);
  const [statusText, setStatusText] = useState("CONNECTING...");
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [activeModelName, setActiveModelName] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState("ppo");
  const [showModelMenu, setShowModelMenu] = useState(false);
  
  const ws = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const hasAutoStarted = useRef(false);

  // Fetch available AI models
  useEffect(() => {
    fetch("/api/v1/models")
      .then(res => res.json())
      .then(data => {
        if (data.models) setAvailableModels(data.models);
      })
      .catch(err => console.error("Could not fetch models", err));
  }, []);

  // Fetch Scenario Meta & Init environment
  useEffect(() => {
    async function initEnv() {
      try {
        // Fetch meta
        const metaRes = await fetch(`/api/v1/scenarios/${scenarioKey}`);
        if(metaRes.ok) {
          setScenarioMeta(await metaRes.json());
        }

        // Reset/Create Env
        const res = await fetch('/api/v1/env/reset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario: scenarioKey })
        });
        
        if (!res.ok) throw new Error("Failed to create environment");
        
        const data = await res.json();
        setEnvId(data.env_id);
        setMaxSteps(data.info?.max_steps || 50);
        
        setLogs([
          { type: 'system', text: `Successfully initialized sandbox [${data.env_id}]` },
          { type: 'system', text: data.info?.task_name ? `Task: ${data.info.task_name}` : '' }
        ]);

      } catch (e) {
        console.error(e);
        setLogs([{ type: 'error', text: 'FATAL: Cannot reach backend server at localhost:8000. Ensure server.py is running.' }]);
        setStatusText("DISCONNECTED");
      }
    }
    
    initEnv();
    
    return () => {
      if (ws.current) ws.current.close();
    };
  }, [scenarioKey]);

  // Handle WebSocket
  useEffect(() => {
    if (!envId) return;

    setStatusText("CONNECTING WS...");
    const socket = new WebSocket(`ws://127.0.0.1:8000/ws/env/${envId}`);
    ws.current = socket;

    socket.onopen = () => {
      setIsReady(true);
      setStatusText("LIVE CONNECTION: STABLE");
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "output") {
        setLogs(prev => [...prev, { type: 'output', text: data.output || " " }]);
        setScore(data.score || 0);
        setStep(data.step || 0);
        if (data.done) {
          setLogs(prev => [...prev, { type: 'system', text: '--- SCENARIO COMPLETED ---' }]);
        }
      } else if (data.type === "input") {
        setLogs(prev => [...prev, { type: 'input', text: data.text }]);
      } else if (data.type === "system") {
        setLogs(prev => [...prev, { type: 'system', text: data.text }]);
      } else if (data.type === "error") {
        setLogs(prev => [...prev, { type: 'error', text: data.message }]);
      }
    };

    socket.onclose = () => {
      setIsReady(false);
      setStatusText("OFFLINE");
    };

    socket.onerror = (e) => {
      console.error("WS Error", e);
      setStatusText("CONNECTION ERROR");
    };

    return () => {
      socket.close();
    };
  }, [envId]);

  // Auto-scroll terminal
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Handle Auto Start Agent
  useEffect(() => {
    if (isReady && autoAgent && !hasAutoStarted.current && envId) {
      hasAutoStarted.current = true;
      startAgent(autoAgent);
    }
  }, [isReady, autoAgent, envId]);

  const startAgent = async (type: string, modelName?: string) => {
    if (!envId) return;
    setActiveAgent(type);
    const mn = modelName || selectedModel;
    setActiveModelName(mn);
    const displayName = availableModels.find(m => m.name === mn)?.display_name || mn.toUpperCase();
    setLogs(prev => [...prev, { type: 'system', text: `INITIATING AUTONOMOUS MODE: [${type.toUpperCase()}] Model: ${displayName}` }]);
    try {
      await fetch(`/api/v1/env/${envId}/agent_run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_type: type, model_name: mn })
      });
    } catch(e) {
      console.error(e);
      setActiveAgent(null);
      setActiveModelName(null);
    }
  };

  const handleCommand = () => {
    if (!command.trim() || !ws.current || !isReady || activeAgent) return;
    
    // Optimistic UI update
    setLogs(prev => [...prev, { type: 'input', text: command }]);
    
    // Send over socket
    ws.current.send(JSON.stringify({
      action: "step",
      command: command
    }));
    
    setCommand("");
  };

  const percentageOptions = {
    // If score is a float 0 to 1
    scorePct: Math.round(score * 100)
  };

  return (
    <div className="flex h-full w-full bg-chaos-dark overflow-hidden p-6 gap-6">
      
      {/* Left Sidebar */}
      <div className="w-[300px] flex flex-col gap-8 shrink-0">
        <div>
          <div className="flex justify-between items-start mb-2">
            <h2 className="text-2xl font-bold leading-tight">{scenarioMeta?.name || "Scenario Loading"}</h2>
            <div className="bg-chaos-cyan/10 border border-chaos-cyan/30 text-chaos-cyan text-[10px] font-mono px-2 py-1 rounded">
              ID:<br/>{envId?.substring(0, 5).toUpperCase() || '...'}
            </div>
          </div>
          <p className="text-chaos-muted text-sm leading-relaxed mt-4">
            {scenarioMeta?.description || "Initialize the simulator and analyze the faults."}
          </p>
        </div>

        {/* Chaos Score Donut */}
        <div className="flex items-center gap-6 p-4 bg-chaos-panel/30 border border-chaos-border rounded-lg">
          <div className="relative w-16 h-16 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path
                className="text-chaos-border"
                strokeWidth="3"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="text-chaos-green transition-all duration-1000"
                strokeWidth="3"
                strokeDasharray={`${percentageOptions.scorePct}, 100`}
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <span className="absolute text-sm font-bold">{percentageOptions.scorePct}%</span>
          </div>
          <div>
            <div className="text-[10px] font-bold text-chaos-muted tracking-widest mb-1 uppercase">Completion Level</div>
            <div className="text-2xl font-bold text-chaos-green">{percentageOptions.scorePct} <span className="text-xs text-chaos-green/50">PTS</span></div>
          </div>
        </div>

        {/* Progress & Objectives */}
        <div className="flex-1">
          <div className="flex justify-between text-xs font-bold text-chaos-muted uppercase tracking-widest mb-6">
            <span>Step Progress</span>
            <span>{step} / {maxSteps}</span>
          </div>
          
          <div className="text-xs font-bold text-chaos-muted uppercase tracking-widest mb-4">Objectives Meta</div>
          <div className="space-y-6">
             <p className="text-xs text-chaos-muted italic">Complete actions in the terminal and verify fixes to achieve 100% completion in this sandbox.</p>
          </div>
        </div>
      </div>

      {/* Main Terminal Window */}
      <div className="flex-1 bg-chaos-panel/40 border border-chaos-border rounded-xl flex flex-col overflow-hidden relative shadow-2xl">
        {/* Terminal Header */}
        <div className="bg-chaos-panel border-b border-chaos-border px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-chaos-red/80"></div>
              <div className="w-3 h-3 rounded-full bg-chaos-muted/50"></div>
              <div className="w-3 h-3 rounded-full bg-chaos-green/80"></div>
            </div>
            <div className="ml-4 flex text-xs font-mono text-chaos-muted space-x-1 border-r border-chaos-border/50 pr-4">
              <span className="text-chaos-text px-2 py-1 bg-chaos-darker rounded border border-chaos-border/50">
                root@chaoslab-{envId || 'init'} ~ (ssh)
              </span>
            </div>
            {/* AGENT INJECTOR */}
            <div className="flex items-center text-xs font-mono pl-1">
              {activeAgent ? (
                <span className="text-chaos-cyan bg-chaos-cyan/10 border border-chaos-cyan/30 px-2 py-1 rounded flex items-center gap-2">
                  <Brain className="w-3 h-3 animate-pulse" /> {activeAgent.toUpperCase()} — {availableModels.find(m => m.name === activeModelName)?.display_name || activeModelName?.toUpperCase() || ''}
                </span>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="text-chaos-muted uppercase ml-2 tracking-widest">Auto-Solve:</span>
                  <button onClick={() => startAgent('llm')} className="text-chaos-muted tracking-widest hover:text-chaos-cyan border border-chaos-border hover:border-chaos-cyan px-2 py-1 rounded transition-colors" title="Auto-Solve with LLM">LLM</button>
                  
                  {/* RL Model Selector Dropdown */}
                  <div className="relative">
                    <button 
                      onClick={() => setShowModelMenu(!showModelMenu)}
                      className="text-chaos-muted tracking-widest hover:text-chaos-green border border-chaos-border hover:border-chaos-green px-2 py-1 rounded transition-colors flex items-center gap-1"
                      title="Auto-Solve with RL Model"
                    >
                      RL <ChevronDown className="w-3 h-3" />
                    </button>
                    {showModelMenu && (
                      <div className="absolute top-full right-0 mt-1 bg-chaos-darker border border-chaos-border rounded-lg shadow-xl z-50 min-w-[220px] overflow-hidden">
                        <div className="px-3 py-2 border-b border-chaos-border text-[10px] text-chaos-muted uppercase tracking-widest">Select AI Model</div>
                        {availableModels.filter(m => m.name !== 'llm').map(m => (
                          <button
                            key={m.name}
                            onClick={() => {
                              setSelectedModel(m.name);
                              setShowModelMenu(false);
                              startAgent('rl', m.name);
                            }}
                            className="w-full text-left px-3 py-2.5 hover:bg-chaos-green/10 transition-colors border-b border-chaos-border/30 last:border-0"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-mono text-chaos-text font-bold">{m.display_name}</span>
                              {m.available ? (
                                <span className="text-[9px] text-chaos-green bg-chaos-green/10 px-1.5 py-0.5 rounded font-mono">READY</span>
                              ) : (
                                <span className="text-[9px] text-chaos-muted bg-chaos-muted/10 px-1.5 py-0.5 rounded font-mono">N/A</span>
                              )}
                            </div>
                            <div className="text-[10px] text-chaos-muted mt-0.5 leading-relaxed">{m.algorithm}</div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Terminal Body */}
        <div className="p-6 font-mono text-sm overflow-y-auto flex-1 flex flex-col" onClick={() => document.getElementById("cli-input")?.focus()}>
          <div className="text-chaos-muted mb-4 opacity-70">
            ChaosLab Live Terminal [Version 2.0.42-STABLE]<br/>
            (c) 2026 ChaosLab System. All rights reserved.
          </div>
          
          {logs.map((log, i) => (
            <div key={i} className="mb-2 whitespace-pre-wrap flex flex-col">
              {log.type === 'input' && (
                <div><span className="text-chaos-green font-bold">root@chaoslab:~$</span> {log.text}</div>
              )}
              {log.type === 'output' && (
                <div className="text-chaos-muted mt-1">{log.text}</div>
              )}
              {log.type === 'error' && (
                <div className="text-chaos-red mt-1">{log.text}</div>
              )}
              {log.type === 'system' && (
                <div className="text-chaos-cyan opacity-80 mt-1 italic">:: {log.text}</div>
              )}
            </div>
          ))}

          <div className="flex items-center gap-2 mt-2" ref={logsEndRef}>
            <span className="text-chaos-green font-bold shrink-0">root@chaoslab:~$</span>
            <input 
              id="cli-input"
              type="text" 
              value={command} 
              onChange={e => setCommand(e.target.value)}
              onKeyDown={e => {
                 if(e.key === 'Enter') handleCommand();
              }}
              className="bg-transparent border-none outline-none flex-1 font-mono text-chaos-text focus:ring-0 disabled:opacity-50"
              autoFocus
              disabled={!isReady || activeAgent !== null}
              autoComplete="off"
            />
          </div>
        </div>

        {/* Terminal Footer */}
        <div className="bg-chaos-panel/80 px-4 py-2 border-t border-chaos-border flex justify-between text-[10px] font-mono uppercase tracking-widest text-chaos-muted">
          <div className="flex gap-6">
            <span className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${isReady ? 'bg-chaos-green animate-pulse-slow' : 'bg-chaos-red'}`}></span> 
              {statusText}
            </span>
            {isReady && <span>LATENCY: &lt;10MS</span>}
          </div>
          <div className="flex gap-6">
            <span>ENCRYPTION: AES-256-GCM</span>
          </div>
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="w-[280px] shrink-0 flex flex-col gap-6 h-full overflow-hidden">
        
        {/* Log History */}
        <div className="bg-chaos-panel/30 border border-chaos-border p-4 rounded-xl flex-1 flex flex-col overflow-hidden">
          <div className="flex justify-between items-center mb-6 shrink-0">
            <h3 className="text-xs font-bold uppercase tracking-widest text-chaos-muted">Command Feed</h3>
            <History className="w-4 h-4 text-chaos-muted" />
          </div>
          
          <div className="space-y-4 overflow-y-auto flex-1 pr-2">
            {[...logs].reverse().filter(l => l.type === 'input').map((log, idx) => (
              <div key={idx} className="flex items-start gap-3 opacity-80">
                <div className="w-1.5 h-1.5 rounded-full bg-chaos-green mt-1.5"></div>
                <div className="flex-1 overflow-hidden">
                  <div className="font-mono text-xs text-chaos-text truncate" title={log.text}>{log.text}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
    </div>
  );
}
