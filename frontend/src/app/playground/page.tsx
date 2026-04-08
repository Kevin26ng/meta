"use client";

import { Activity, CircleCheck, Circle, Search, History, CheckCircle2, Brain, ChevronDown, Send, Lightbulb, Zap } from "lucide-react";
import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";

type LogEntry = {
  type: 'input' | 'output' | 'error' | 'system';
  text: string;
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  suggested_command?: string;
  timestamp: number;
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
  
  // Chat panel state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [showChatPanel, setShowChatPanel] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  const llmModel = availableModels.find(m => m.name === "llm");
  const llmAvailable = Boolean(llmModel?.available);
  
  const ws = useRef<WebSocket | null>(null);
  const wsIntentionalClose = useRef(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const hasAutoStarted = useRef(false);

  // Fetch available AI models
  useEffect(() => {
    fetch("/api/v1/models")
      .then(res => res.json())
      .then(data => {
        if (data.models) setAvailableModels(data.models);
      })
        .catch(err => console.warn("Could not fetch models", err));
  }, []);

  // Fetch Scenario Meta & Init environment
  useEffect(() => {
    async function initEnv() {
      try {
        const [metaRes, res] = await Promise.all([
          fetch(`/api/v1/scenarios/${scenarioKey}`),
          fetch('/api/v1/env/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario: scenarioKey })
          })
        ]);

        if (metaRes.ok) {
          setScenarioMeta(await metaRes.json());
        }

        if (!res.ok) throw new Error("Failed to create environment");

        const data = await res.json();
        setEnvId(data.env_id);
        setMaxSteps(data.info?.max_steps || 50);
        
        setLogs([
          { type: 'system', text: `Successfully initialized sandbox [${data.env_id}]` },
          { type: 'system', text: data.info?.task_name ? `Task: ${data.info.task_name}` : '' }
        ]);

      } catch (e) {
        console.warn("Playground init failed", e);
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
    wsIntentionalClose.current = false;
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${wsProtocol}://${window.location.host}/ws/env/${envId}`);
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

    socket.onerror = () => {
      if (wsIntentionalClose.current) return;
      console.warn("WS connection issue detected");
      setStatusText("CONNECTION ERROR");
    };

    return () => {
      wsIntentionalClose.current = true;
      socket.close();
    };
  }, [envId]);

  // Auto-scroll terminal
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Auto-scroll chat panel
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

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
    const mn = type === 'llm' ? '' : (modelName || selectedModel);
    setActiveModelName(type === 'llm' ? 'llm' : mn);
    const displayName = type === 'llm'
      ? (availableModels.find(m => m.name === 'llm')?.display_name || 'OpenAI LLM Copilot')
      : (availableModels.find(m => m.name === mn)?.display_name || mn.toUpperCase());
    setLogs(prev => [...prev, { type: 'system', text: `INITIATING AUTONOMOUS MODE: [${type.toUpperCase()}] Model: ${displayName}` }]);
    try {
      await fetch(`/api/v1/env/${envId}/agent_run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_type: type, model_name: mn })
      });
    } catch(e) {
      console.warn("Agent start failed", e);
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

  const handleChatSubmit = async () => {
    if (!chatInput.trim() || !envId || isChatLoading) return;
    
    const userQuery = chatInput;
    setChatInput("");
    
    // Add user message to chat
    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}-user`,
      role: 'user',
      content: userQuery,
      timestamp: Date.now(),
    };
    setChatMessages(prev => [...prev, userMsg]);
    
    setIsChatLoading(true);
    try {
      const res = await fetch(`/api/v1/chat/${envId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userQuery })
      });
      
      if (!res.ok) throw new Error("Chat request failed");
      
      const data = await res.json();
      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now()}-assistant`,
        role: 'assistant',
        content: data.response,
        suggested_command: data.suggested_command,
        timestamp: Date.now(),
      };
      setChatMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      console.warn("Chat error:", err);
      setChatMessages(prev => [...prev, {
        id: `msg-${Date.now()}-error`,
        role: 'assistant',
        content: "Error: Could not get response. Check LLM configuration.",
        timestamp: Date.now(),
      }]);
    } finally {
      setIsChatLoading(false);
    }
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
                  <button
                    onClick={() => llmAvailable && startAgent('llm')}
                    disabled={!llmAvailable}
                    className={`text-chaos-muted tracking-widest border border-chaos-border px-2 py-1 rounded transition-colors ${llmAvailable ? 'hover:text-chaos-cyan hover:border-chaos-cyan' : 'opacity-50 cursor-not-allowed'}`}
                    title={llmAvailable ? 'Auto-Solve with LLM' : 'Configure API_BASE_URL, MODEL_NAME, and HF_TOKEN to enable the LLM'}
                  >LLM</button>
                  
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
        
        {/* AI Chat Assistant */}
        <div className="bg-chaos-panel/30 border border-chaos-border p-4 rounded-xl flex flex-col overflow-hidden flex-1">
          <div className="flex justify-between items-center mb-4 shrink-0">
            <h3 className="text-xs font-bold uppercase tracking-widest text-chaos-muted flex items-center gap-2">
              <Lightbulb className="w-3.5 h-3.5 text-chaos-cyan" /> AI Assistant
            </h3>
            {!llmAvailable && (
              <span className="text-[8px] text-chaos-muted bg-chaos-muted/20 px-1.5 py-0.5 rounded">OFFLINE</span>
            )}
          </div>
          
          {/* Chat messages */}
          <div className="flex-1 overflow-y-auto mb-3 space-y-3 pr-2">
            {chatMessages.length === 0 && (
              <div className="text-[11px] text-chaos-muted/60 italic">
                Ask the LLM for help with the current task. Get suggestions for the next command to run.
              </div>
            )}
            {chatMessages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[90%] rounded-lg px-3 py-2 text-[11px] leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-chaos-cyan/20 border border-chaos-cyan/30 text-chaos-cyan'
                    : 'bg-chaos-green/10 border border-chaos-green/20 text-chaos-text'
                }`}>
                  <div>{msg.content}</div>
                  {msg.suggested_command && (
                    <div className="mt-2 pt-2 border-t border-current/20">
                      <div className="text-[10px] opacity-70 mb-1.5">Suggested command:</div>
                      <button
                        onClick={() => {
                          setCommand(msg.suggested_command!);
                          setTimeout(() => document.getElementById("cli-input")?.focus(), 0);
                        }}
                        className="w-full text-left font-mono text-[10px] bg-chaos-darker/50 px-2 py-1 rounded border border-chaos-border/50 hover:border-chaos-green/50 transition-colors"
                        title="Click to paste command"
                      >
                        <Zap className="w-2.5 h-2.5 inline mr-1" />
                        {msg.suggested_command.substring(0, 35)}
                        {msg.suggested_command.length > 35 ? "..." : ""}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isChatLoading && (
              <div className="flex justify-start">
                <div className="bg-chaos-green/10 border border-chaos-green/20 rounded-lg px-3 py-2">
                  <div className="text-[11px] text-chaos-text animate-pulse">Thinking...</div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          
          {/* Chat input */}
          <div className="flex gap-2 shrink-0">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleChatSubmit()}
              placeholder={llmAvailable ? "Ask for a command..." : "LLM offline"}
              disabled={!llmAvailable || isChatLoading}
              className="flex-1 bg-chaos-darker/50 border border-chaos-border/50 rounded px-2 py-1.5 text-[11px] text-chaos-text placeholder-chaos-muted/50 focus:outline-none focus:border-chaos-cyan/50 disabled:opacity-50"
            />
            <button
              onClick={handleChatSubmit}
              disabled={!llmAvailable || isChatLoading || !chatInput.trim()}
              className="bg-chaos-cyan/10 border border-chaos-cyan/30 text-chaos-cyan p-1.5 rounded hover:bg-chaos-cyan/20 disabled:opacity-30 transition-colors"
              title="Send to LLM"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
      
    </div>
  );
}
