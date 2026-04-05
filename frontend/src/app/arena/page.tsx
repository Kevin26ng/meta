"use client";

import { Zap, Server, ChevronDown, CheckCircle2, ShieldAlert, Cpu, Brain } from "lucide-react";
import { useState, useEffect } from "react";

export default function ArenaPage() {
  const [scenarios, setScenarios] = useState<Record<string, any>>({});
  const [activeScenario, setActiveScenario] = useState<string>("log_analysis");
  
  const [commandsA, setCommandsA] = useState("cat /var/log/app.log\ngrep 500 /var/log/app.log");
  const [commandsB, setCommandsB] = useState("ls -la\ncat /var/log/app.log");
  const [typeA, setTypeA] = useState<string>("script");
  const [typeB, setTypeB] = useState<string>("rl");
  const [modelA, setModelA] = useState<string>("ppo");
  const [modelB, setModelB] = useState<string>("heuristic");
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  
  const [isRacing, setIsRacing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/v1/scenarios")
      .then(res => res.json())
      .then(data => {
        if(data.scenarios) {
          setScenarios(data.scenarios);
          if (!data.scenarios["log_analysis"]) {
             setActiveScenario(Object.keys(data.scenarios)[0]);
          }
        }
      })
      .catch(err => {
        console.error("Could not fetch scenarios", err);
        setError("Cannot reach backend. Is server.py running?");
      });

    // Fetch available AI models
    fetch("/api/v1/models")
      .then(res => res.json())
      .then(data => {
        if (data.models) setAvailableModels(data.models);
      })
      .catch(err => console.error("Could not fetch models", err));
  }, []);

  const runRace = async () => {
    setIsRacing(true);
    setError(null);
    setResults(null);
    try {
      const res = await fetch("/api/v1/arena/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario: activeScenario,
          commands_a: commandsA.split("\n").filter(c => c.trim()),
          commands_b: commandsB.split("\n").filter(c => c.trim()),
          label_a: "AGENT_ALPHA",
          label_b: "AGENT_BETA",
          type_a: typeA,
          type_b: typeB,
          model_a: modelA,
          model_b: modelB
        })
      });
      if(!res.ok) {
        const errBody = await res.text();
        throw new Error(`Server returned ${res.status}: ${errBody}`);
      }
      setResults(await res.json());
    } catch(e: any) {
      console.error("Arena error:", e);
      setError(e.message || "Unknown error running arena");
    }
    setIsRacing(false);
  };

  const alphaScore = results ? Math.round(results.results["AGENT_ALPHA"].final_score * 100) : 0;
  const betaScore = results ? Math.round(results.results["AGENT_BETA"].final_score * 100) : 0;
  const winner = results ? results.winner : "PENDING";
  
  const alphaHistory = results ? results.results["AGENT_ALPHA"].history : [];
  const betaHistory = results ? results.results["AGENT_BETA"].history : [];

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 pb-20">
      
      {/* Header */}
      <div className="mb-10">
        <div className="text-xs font-mono text-chaos-green mb-4">root@chaoslab-terminal / ARENA / AGENT_DUEL_V3.4</div>
        <h1 className="text-5xl font-extrabold mb-4 italic tracking-tighter" style={{ fontFamily: 'var(--font-serif)' }}>Battle Scenario</h1>
        <p className="text-chaos-muted max-w-2xl text-base">
          Compare user or autonomous scripts against high-stakes environments. Define identical testing parameters and observe how different instruction sets handle cascading service disruptions.
        </p>
      </div>

      {/* Active Environment Banner */}
      <div className="bg-chaos-panel/50 border border-chaos-border rounded-lg p-5 flex justify-between items-center mb-10">
        <div>
          <div className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-1">Active Environment Overlay</div>
          <select 
            value={activeScenario} 
            onChange={e => setActiveScenario(e.target.value)}
            className="bg-transparent border-none outline-none text-lg font-mono text-chaos-text cursor-pointer appearance-none uppercase"
          >
            {Object.keys(scenarios).map(key => (
               <option key={key} value={key} className="bg-chaos-dark">{scenarios[key].name}</option>
            ))}
            {Object.keys(scenarios).length === 0 && <option value="log_analysis">Loading Scenarios...</option>}
          </select>
        </div>
        <div className="flex items-center gap-2 text-chaos-green font-mono text-sm">
           <span className="w-2 h-2 rounded-full bg-chaos-green animate-pulse"></span> READY
        </div>
      </div>

      {/* Agents Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
        {/* AGENT ALPHA */}
        <div className="bg-chaos-panel/30 border border-chaos-border rounded-xl p-6 relative overflow-hidden group hover:border-chaos-green/30 transition-colors flex flex-col">
          <div className="absolute top-0 right-0 text-[120px] font-bold text-chaos-darker leading-none select-none pointer-events-none group-hover:text-chaos-green/5 transition-colors">A</div>
          
          <div className="flex items-center gap-3 mb-6 relative z-10">
            <span className="w-3 h-3 rounded-full bg-chaos-green animate-pulse"></span>
            <h2 className="text-xl font-bold font-mono tracking-wider text-chaos-green">AGENT_ALPHA</h2>
          </div>
          
          <div className="mb-4 relative z-10">
            <label className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-2 block">Behavior Matrix</label>
            <select 
              value={typeA} 
              onChange={e => setTypeA(e.target.value)}
              className="w-full bg-chaos-darker border border-chaos-border rounded p-2 text-sm font-mono text-chaos-text outline-none focus:border-chaos-green"
            >
              <option value="script">Hardcoded Script</option>
              <option value="llm">LLM Autonomous</option>
              <option value="rl">RL Autonomous</option>
            </select>
          </div>

          {typeA === "script" ? (
            <textarea 
              className="w-full bg-chaos-dark rounded flex-1 p-4 border border-chaos-border min-h-[160px] mb-4 font-mono text-sm text-chaos-text relative z-10 box-border focus:border-chaos-green focus:outline-none"
              placeholder="Enter shell commands, one per line..."
              value={commandsA}
              onChange={e => setCommandsA(e.target.value)}
            />
          ) : typeA === "rl" ? (
            <div className="flex-1 min-h-[160px] border border-chaos-border bg-chaos-darker rounded mb-4 relative z-10 flex flex-col items-center justify-center text-center p-6 bg-texture-stripes">
               <Brain className="w-12 h-12 text-chaos-green mb-4 opacity-50" />
               <h3 className="font-mono text-chaos-green font-bold mb-2">RL AUTONOMOUS MODE</h3>
               <p className="text-xs text-chaos-muted tracking-wider leading-relaxed mb-4">
                 Model: {availableModels.find(m => m.name === modelA)?.display_name || modelA.toUpperCase()}
               </p>
               <select
                 value={modelA}
                 onChange={e => setModelA(e.target.value)}
                 className="bg-chaos-dark border border-chaos-border rounded px-3 py-1.5 text-xs font-mono text-chaos-text outline-none focus:border-chaos-green"
               >
                 {availableModels.map(m => (
                   <option key={m.name} value={m.name} className="bg-chaos-dark">{m.display_name} {m.available ? '✓' : '—'}</option>
                 ))}
               </select>
            </div>
          ) : (
            <div className="flex-1 min-h-[160px] border border-chaos-border bg-chaos-darker rounded mb-4 relative z-10 flex flex-col items-center justify-center text-center p-6 bg-texture-stripes">
               <Brain className="w-12 h-12 text-chaos-green mb-4 opacity-50" />
               <h3 className="font-mono text-chaos-green font-bold mb-2">AUTONOMOUS MODE ACTIVE</h3>
               <p className="text-xs text-chaos-muted tracking-wider leading-relaxed">
                 LLM Neural Net will dynamically respond to terminal output.
               </p>
            </div>
          )}
        </div>

        {/* AGENT BETA */}
        <div className="bg-chaos-panel/30 border border-chaos-border rounded-xl p-6 relative overflow-hidden group hover:border-chaos-red/30 transition-colors flex flex-col">
          <div className="absolute top-0 right-0 text-[120px] font-bold text-chaos-darker leading-none select-none pointer-events-none group-hover:text-chaos-red/5 transition-colors">B</div>
          
          <div className="flex items-center gap-3 mb-6 relative z-10">
            <span className="w-3 h-3 rounded-full bg-chaos-red"></span>
            <h2 className="text-xl font-bold font-mono tracking-wider text-chaos-red">AGENT_BETA</h2>
          </div>
          
          <div className="mb-4 relative z-10">
            <label className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-2 block">Behavior Matrix</label>
            <select 
              value={typeB} 
              onChange={e => setTypeB(e.target.value)}
              className="w-full bg-chaos-darker border border-chaos-border rounded p-2 text-sm font-mono text-chaos-text outline-none focus:border-chaos-red"
            >
              <option value="script">Hardcoded Script</option>
              <option value="llm">LLM Autonomous</option>
              <option value="rl">RL Autonomous</option>
            </select>
          </div>

          {typeB === "script" ? (
            <textarea 
              className="w-full bg-chaos-dark rounded flex-1 p-4 border border-chaos-border min-h-[160px] mb-4 font-mono text-sm text-chaos-text relative z-10 box-border focus:border-chaos-red focus:outline-none"
              placeholder="Enter shell commands, one per line..."
              value={commandsB}
              onChange={e => setCommandsB(e.target.value)}
            />
          ) : typeB === "rl" ? (
            <div className="flex-1 min-h-[160px] border border-chaos-border bg-chaos-darker rounded mb-4 relative z-10 flex flex-col items-center justify-center text-center p-6 bg-texture-stripes">
               <Brain className="w-12 h-12 text-chaos-red mb-4 opacity-50" />
               <h3 className="font-mono text-chaos-red font-bold mb-2">RL AUTONOMOUS MODE</h3>
               <p className="text-xs text-chaos-muted tracking-wider leading-relaxed mb-4">
                 Model: {availableModels.find(m => m.name === modelB)?.display_name || modelB.toUpperCase()}
               </p>
               <select
                 value={modelB}
                 onChange={e => setModelB(e.target.value)}
                 className="bg-chaos-dark border border-chaos-border rounded px-3 py-1.5 text-xs font-mono text-chaos-text outline-none focus:border-chaos-red"
               >
                 {availableModels.map(m => (
                   <option key={m.name} value={m.name} className="bg-chaos-dark">{m.display_name} {m.available ? '✓' : '—'}</option>
                 ))}
               </select>
            </div>
          ) : (
            <div className="flex-1 min-h-[160px] border border-chaos-border bg-chaos-darker rounded mb-4 relative z-10 flex flex-col items-center justify-center text-center p-6 bg-texture-stripes">
               <Brain className="w-12 h-12 text-chaos-red mb-4 opacity-50" />
               <h3 className="font-mono text-chaos-red font-bold mb-2">AUTONOMOUS MODE ACTIVE</h3>
               <p className="text-xs text-chaos-muted tracking-wider leading-relaxed">
                 LLM Neural Net will dynamically respond to terminal output.
               </p>
            </div>
          )}
        </div>
      </div>

      {/* Central Action */}
      <div className="flex justify-center mb-16 relative">
        <div className="absolute top-1/2 left-0 right-0 h-px bg-chaos-border -translate-y-1/2 -z-10"></div>
        <button 
          onClick={runRace}
          disabled={isRacing}
          className="bg-chaos-green text-chaos-dark disabled:bg-chaos-muted disabled:text-chaos-darker font-extrabold text-lg px-12 py-4 rounded hover:bg-chaos-green/90 transition-all uppercase tracking-widest flex items-center gap-3 shadow-[0_0_30px_rgba(57,255,20,0.3)] hover:shadow-[0_0_50px_rgba(57,255,20,0.5)] transform hover:scale-105"
        >
          {isRacing ? "SIMULATING..." : <>RUN RACE <Zap className="w-5 h-5 fill-current" /></>}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-chaos-red/10 border border-chaos-red/30 rounded-lg p-4 mb-8 text-center">
          <p className="text-chaos-red font-mono text-sm">{error}</p>
        </div>
      )}

      {/* Results Section */}
      {results && (
      <div className="bg-chaos-panel/40 border border-chaos-border rounded-xl p-8 mb-12 animate-in fade-in slide-in-from-bottom-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-12">
          
          {/* Alpha Score Ring */}
          <div className="flex flex-col items-center gap-4">
            <div className={`relative w-32 h-32 flex items-center justify-center ${winner === 'AGENT_ALPHA' ? '' : 'opacity-80'}`}>
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                <path className="text-chaos-border" strokeWidth="3" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path className="text-chaos-cyan shadow-[0_0_10px_rgba(0,255,255,0.5)]" strokeWidth="3" strokeDasharray={`${alphaScore}, 100`} stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-3xl font-bold text-chaos-text">{alphaScore}%</span>
                <span className="text-[10px] text-chaos-muted uppercase tracking-widest mt-1">Score</span>
              </div>
            </div>
            <div className="font-bold font-mono text-sm tracking-wider">Alpha Score</div>
          </div>

          {/* Central Metrics */}
          <div className={`flex-1 bg-chaos-darker border-l-4 rounded-lg p-6 relative overflow-hidden ${winner === 'AGENT_ALPHA' ? 'border-chaos-cyan' : winner === 'AGENT_BETA' ? 'border-chaos-red' : 'border-chaos-muted'}`}>
            <div className="absolute right-6 top-6 text-right">
              {winner === 'tie' ? (
                 <div className="text-xl font-bold text-chaos-muted">STALEMATE</div>
              ) : (
                 <div className="text-xl font-bold text-chaos-green">VICTORY</div>
              )}
            </div>
            
            <div className="font-mono text-xs text-chaos-text bg-chaos-border/50 border border-chaos-border px-2 py-1 rounded w-max mb-3 uppercase tracking-widest">
              {winner === 'tie' ? 'Tie Game' : 'Winner'}
            </div>
            <h3 className="text-2xl font-bold font-mono tracking-wider mb-6">{winner}</h3>
            
            <div className="space-y-4 font-mono text-sm">
              <div className="flex justify-between items-center border-b border-chaos-border pb-2">
                <span className="text-chaos-muted">Commands Sent (A)</span>
                <span className="text-chaos-text">{results.results["AGENT_ALPHA"].steps_used}</span>
              </div>
              <div className="flex justify-between items-center border-b border-chaos-border pb-2">
                <span className="text-chaos-muted">Commands Sent (B)</span>
                <span className="text-chaos-text">{results.results["AGENT_BETA"].steps_used}</span>
              </div>
            </div>
          </div>

          {/* Beta Score Ring */}
          <div className="flex flex-col items-center gap-4">
            <div className={`relative w-32 h-32 flex items-center justify-center ${winner === 'AGENT_BETA' ? '' : 'opacity-80'}`}>
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                <path className="text-chaos-border" strokeWidth="3" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path className="text-chaos-red" strokeWidth="3" strokeDasharray={`${betaScore}, 100`} stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-3xl font-bold text-chaos-text">{betaScore}%</span>
                <span className="text-[10px] text-chaos-muted uppercase tracking-widest mt-1">Score</span>
              </div>
            </div>
            <div className="font-bold font-mono text-sm tracking-wider">Beta Score</div>
          </div>
        </div>
      </div>
      )}

      {/* Execution Timelines */}
      {results && (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 animate-in fade-in slide-in-from-bottom-12">
        <div>
          <h4 className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-6 font-mono border-b border-chaos-border pb-2">Execution Timeline: Alpha</h4>
          <div className="relative pl-6 space-y-8 before:absolute before:inset-0 before:left-[11px] before:w-px before:bg-chaos-border before:z-0">
            
            {alphaHistory.map((h: any, idx: number) => (
             <div key={idx} className="relative z-10 flex gap-4">
               <div className={`w-6 h-6 rounded-full border flex items-center justify-center shrink-0 mt-1 ${h.exit_code === 0 ? 'bg-chaos-cyan/20 border-chaos-cyan text-chaos-cyan' : 'bg-chaos-red/20 border-chaos-red text-chaos-red'}`}>
                 {h.exit_code === 0 ? <CheckCircle2 className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3"/>}
               </div>
               <div>
                  <div className="text-[10px] text-chaos-muted font-mono uppercase tracking-widest mb-1">STEP {idx + 1}</div>
                  <div className="text-sm font-bold text-chaos-text mb-1 font-mono">{h.command}</div>
                  <div className="text-xs text-chaos-muted leading-relaxed">Score Impact: {Math.round(h.score * 100)}%</div>
               </div>
            </div>
            ))}
            
          </div>
        </div>
        
        <div>
          <h4 className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-6 font-mono border-b border-chaos-border pb-2">Execution Timeline: Beta</h4>
          <div className="relative pl-6 space-y-8 before:absolute before:inset-0 before:left-[11px] before:w-px before:bg-chaos-border before:z-0">
            
            {betaHistory.map((h: any, idx: number) => (
             <div key={idx} className="relative z-10 flex gap-4 opacity-80">
               <div className={`w-6 h-6 rounded-full border flex items-center justify-center shrink-0 mt-1 ${h.exit_code === 0 ? 'bg-chaos-red/20 border-chaos-red text-chaos-red' : 'bg-chaos-darker border-chaos-border text-chaos-muted'}`}>
                 {h.exit_code === 0 ? <CheckCircle2 className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3"/>}
               </div>
               <div>
                  <div className="text-[10px] text-chaos-muted font-mono uppercase tracking-widest mb-1">STEP {idx + 1}</div>
                  <div className="text-sm font-bold text-chaos-text mb-1 font-mono">{h.command}</div>
                  <div className="text-xs text-chaos-muted leading-relaxed">Score Impact: {Math.round(h.score * 100)}%</div>
               </div>
            </div>
            ))}

          </div>
        </div>
      </div>
      )}

    </div>
  );
}
