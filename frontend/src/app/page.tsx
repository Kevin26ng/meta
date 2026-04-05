import { Activity, Skull, HardDrive, Globe, Cpu, Server, Filter, SlidersHorizontal, Settings } from "lucide-react";
import Link from "next/link";

interface ScenarioItem {
  id: string;
  name: string;
  difficulty: string;
  description: string;
  objectives_count: number;
  max_steps: number;
}

// Ensure the page gets fresh data (no static optimization for the api call)
export const dynamic = "force-dynamic";

export default async function HubPage() {
  let scenarios: Record<string, any> = {};
  
  try {
    const res = await fetch("http://127.0.0.1:8000/api/v1/scenarios", { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      scenarios = data.scenarios || {};
    }
  } catch (e) {
    console.error("Failed to fetch scenarios from backend:", e);
    // Silent fail to empty object if backend is not running yet
  }

  // Convert the dictionary to an array for mapping
  const scenarioList = Object.keys(scenarios).map(key => ({
    id: key,
    ...scenarios[key]
  }));

  // Helper to map backend difficulty to UI tags
  const getStatus = (diff: string) => {
    switch (diff) {
      case 'easy': return { label: "SAFE", color: "text-chaos-green bg-chaos-green/10 border-chaos-green/20" };
      case 'medium': return { label: "MEDIUM", color: "text-chaos-cyan bg-chaos-cyan/10 border-chaos-cyan/20" };
      case 'hard': return { label: "CRITICAL", color: "text-chaos-red bg-chaos-red/10 border-chaos-red/20" };
      case 'expert': return { label: "EXPERT", color: "text-chaos-red bg-chaos-red/10 border-chaos-red/50" };
      default: return { label: "UNKNOWN", color: "text-chaos-muted bg-chaos-darker border-chaos-border" };
    }
  };

  // Helper to get an icon based on scenario name/type
  const getIcon = (id: string, diff: string) => {
    const props = { className: `w-5 h-5 ${diff === 'easy' ? 'text-chaos-green' : diff === 'medium' ? 'text-chaos-cyan' : 'text-chaos-red'}` };
    if (id.includes('disk')) return <HardDrive {...props} />;
    if (id.includes('network')) return <Globe {...props} />;
    if (id.includes('mem')) return <Activity {...props} />;
    if (id.includes('process') || id.includes('db')) return <Server {...props} />;
    if (id.includes('security')) return <Skull {...props} />;
    return <Cpu {...props} />;
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 pb-20">
      
      {/* Header section */}
      <div className="flex justify-between items-start mb-10">
        <div>
          <h1 className="text-4xl font-bold mb-3 tracking-tight">Chaos<span className="text-chaos-green/80">Hub</span></h1>
          <p className="text-chaos-muted max-w-2xl text-lg">
            Central nervous system for system resilience. Browse production-hardened tools and scenarios to stress test your architecture.
          </p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 bg-chaos-panel border border-chaos-border px-4 py-2 rounded text-sm hover:border-chaos-muted transition-colors">
            <Filter className="w-4 h-4" /> Filter
          </button>
          <button className="flex items-center gap-2 bg-chaos-panel border border-chaos-border px-4 py-2 rounded text-sm hover:border-chaos-muted transition-colors">
            <SlidersHorizontal className="w-4 h-4" /> Complexity
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-8 border-b border-chaos-border mb-8">
        <button className="text-chaos-green border-b-2 border-chaos-green pb-3 font-medium">Scenarios</button>
        <button className="text-chaos-muted hover:text-chaos-text pb-3 font-medium transition-colors">Integrations</button>
      </div>

      {scenarioList.length === 0 && (
        <div className="text-center py-12 bg-chaos-panel/30 border border-chaos-border rounded-lg mb-12">
          <Activity className="w-10 h-10 text-chaos-muted mx-auto mb-4 animate-pulse" />
          <h3 className="text-lg font-bold text-chaos-text">Waiting for Backend</h3>
          <p className="text-chaos-muted">Run the Python backend to view live scenarios.</p>
        </div>
      )}

      {/* Grid of Scenarios */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        {scenarioList.map((tool) => {
          const status = getStatus(tool.difficulty);
          return (
            <Link key={tool.id} href={`/playground?scenario=${tool.id}`}>
              <div className="h-full group flex flex-col bg-chaos-panel/50 border border-chaos-border hover:border-chaos-green/50 rounded-lg p-6 transition-all hover:bg-chaos-panel hover:shadow-[0_0_20px_rgba(57,255,20,0.05)] cursor-pointer">
                <div className="flex justify-between items-start mb-4">
                  <div className="w-10 h-10 rounded-md bg-chaos-darker flex items-center justify-center border border-chaos-border group-hover:border-chaos-green/30 transition-colors">
                    {getIcon(tool.id, tool.difficulty)}
                  </div>
                  <span className={`text-[10px] font-bold px-2 py-1 rounded border ${status.color}`}>
                    {status.label}
                  </span>
                </div>
                <h3 className="text-lg font-bold mb-2">{tool.name}</h3>
                <p className="text-chaos-muted text-sm flex-1 mb-6 leading-relaxed line-clamp-3">
                  {tool.description}
                </p>
                <div className="flex flex-wrap gap-2 text-[10px] font-mono text-chaos-muted">
                  <span className="bg-chaos-darker px-2 py-1 rounded border border-chaos-border">
                    {tool.objectives_count} OBJS
                  </span>
                  <span className="bg-chaos-darker px-2 py-1 rounded border border-chaos-border">
                    MAX {tool.max_steps} STEPS
                  </span>
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {/* Featured Scenario Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-chaos-panel/30 border border-chaos-border rounded-lg p-8 relative overflow-hidden flex flex-col justify-center">
          <div className="absolute top-0 right-0 w-64 h-64 bg-chaos-green/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/4 pointer-events-none" />
          
          <div className="bg-chaos-green/10 text-chaos-green text-xs font-bold px-3 py-1 rounded w-max tracking-widest mb-4">
            FEATURED SCENARIO
          </div>
          <h2 className="text-3xl font-bold mb-4">Full Incident Response</h2>
          <p className="text-chaos-muted max-w-xl mb-8 leading-relaxed">
            The ultimate challenge. A cascading failure involving database crash, disk overfilling, cron breakdown, and a concurrent security incident. Triage, prioritize, and restore all systems to 100% functionality.
          </p>
          <div className="flex items-center gap-6">
            <Link href="/playground?scenario=full_incident">
              <button className="bg-chaos-green text-chaos-dark font-bold px-6 py-3 rounded hover:bg-chaos-green/90 transition-colors">
                Initialize Sandbox
              </button>
            </Link>
          </div>
          <Settings className="absolute right-8 bottom-8 w-32 h-32 text-chaos-border/50 -rotate-45" />
        </div>
        
        <div className="bg-chaos-panel/30 border border-chaos-border rounded-lg p-8">
          <h3 className="text-lg font-bold mb-6">Technical Specs</h3>
          <div className="space-y-4 font-mono text-sm">
            <div className="flex justify-between border-b border-chaos-border/50 pb-2">
              <span className="text-chaos-muted">Complexity</span>
              <span className="text-chaos-red">Expert</span>
            </div>
            <div className="flex justify-between border-b border-chaos-border/50 pb-2">
              <span className="text-chaos-muted">Max Steps</span>
              <span className="text-chaos-cyan">100</span>
            </div>
            <div className="flex justify-between pb-2">
              <span className="text-chaos-muted">Grading Logic</span>
              <span className="text-chaos-green">6 Objectives</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
