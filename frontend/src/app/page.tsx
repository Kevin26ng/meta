"use client";

import { Activity, Skull, HardDrive, Globe, Cpu, Server, SlidersHorizontal, Settings, Search, X, ChevronDown } from "lucide-react";
import Link from "next/link";
import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";

interface ScenarioItem {
  id: string;
  name: string;
  difficulty: string;
  description: string;
  objectives_count: number;
  max_steps: number;
}

export default function HubPage() {
  const searchParams = useSearchParams();
  const searchQuery = searchParams.get("search") || "";

  const [scenarios, setScenarios] = useState<Record<string, any>>({});
  const [complexityFilter, setComplexityFilter] = useState<string | null>(null);
  const [complexityOpen, setComplexityOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/v1/scenarios")
      .then(r => r.json())
      .then(data => {
        setScenarios(data.scenarios || {});
      })
      .catch(console.error);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setComplexityOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Convert the dictionary to an array for mapping
  const scenarioList = Object.keys(scenarios).map(key => ({
    id: key,
    ...scenarios[key]
  }));

  // Filter scenarios by search (from Navbar) and complexity
  const filteredScenarios = scenarioList.filter((s) => {
    const matchesSearch =
      !searchQuery ||
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.id.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesComplexity =
      !complexityFilter || s.difficulty === complexityFilter;

    return matchesSearch && matchesComplexity;
  });

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

  const complexityOptions = [
    { value: null, label: "All Levels", dotColor: "bg-chaos-muted" },
    { value: "easy", label: "Safe", dotColor: "bg-chaos-green" },
    { value: "medium", label: "Medium", dotColor: "bg-chaos-cyan" },
    { value: "hard", label: "Critical", dotColor: "bg-chaos-red" },
    { value: "expert", label: "Expert", dotColor: "bg-chaos-red" },
  ];

  const currentComplexityLabel = complexityOptions.find(o => o.value === complexityFilter)?.label || "All Levels";

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
        <div className="flex gap-3 items-center">
          {/* Complexity Dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button
              id="complexity-filter"
              onClick={() => setComplexityOpen(!complexityOpen)}
              className={`flex items-center gap-2 bg-chaos-panel border px-4 py-2 rounded text-sm transition-all ${
                complexityFilter
                  ? "border-chaos-green/40 text-chaos-green shadow-[0_0_10px_rgba(57,255,20,0.08)]"
                  : "border-chaos-border hover:border-chaos-muted text-chaos-text"
              }`}
            >
              <SlidersHorizontal className="w-4 h-4" />
              <span>{currentComplexityLabel}</span>
              <ChevronDown className={`w-3.5 h-3.5 text-chaos-muted transition-transform ${complexityOpen ? 'rotate-180' : ''}`} />
            </button>

            {complexityOpen && (
              <div className="absolute top-full right-0 mt-2 w-[180px] bg-chaos-panel border border-chaos-border rounded-lg shadow-xl z-50 overflow-hidden">
                {complexityOptions.map((opt) => (
                  <button
                    key={opt.label}
                    onClick={() => { setComplexityFilter(opt.value); setComplexityOpen(false); }}
                    className={`w-full text-left px-4 py-2.5 text-sm flex items-center gap-3 transition-colors ${
                      complexityFilter === opt.value
                        ? "bg-chaos-darker text-chaos-text"
                        : "hover:bg-chaos-darker/60 text-chaos-muted hover:text-chaos-text"
                    }`}
                  >
                    <span className={`w-2 h-2 rounded-full ${opt.dotColor} ${opt.value === "expert" ? "animate-pulse" : ""}`} />
                    <span className="font-medium">{opt.label}</span>
                    {complexityFilter === opt.value && (
                      <span className="ml-auto text-chaos-green text-xs">✓</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Active Filters */}
      {(searchQuery || complexityFilter) && (
        <div className="flex items-center gap-3 mb-6">
          <span className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest">Active Filters:</span>
          {searchQuery && (
            <span className="flex items-center gap-1.5 text-xs bg-chaos-panel border border-chaos-border rounded px-2.5 py-1 text-chaos-text">
              <Search className="w-3 h-3 text-chaos-muted" />
              &quot;{searchQuery}&quot;
            </span>
          )}
          {complexityFilter && (
            <span className={`flex items-center gap-1.5 text-xs rounded px-2.5 py-1 border ${getStatus(complexityFilter).color}`}>
              {getStatus(complexityFilter).label}
              <button onClick={() => setComplexityFilter(null)} className="hover:opacity-70 transition-opacity ml-1">
                <X className="w-3 h-3" />
              </button>
            </span>
          )}
          {complexityFilter && (
            <button
              onClick={() => setComplexityFilter(null)}
              className="text-[10px] text-chaos-muted hover:text-chaos-red transition-colors uppercase tracking-widest ml-2"
            >
              Clear All
            </button>
          )}
        </div>
      )}

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

      {scenarioList.length > 0 && filteredScenarios.length === 0 && (
        <div className="text-center py-12 bg-chaos-panel/30 border border-chaos-border rounded-lg mb-12">
          <Search className="w-10 h-10 text-chaos-muted mx-auto mb-4" />
          <h3 className="text-lg font-bold text-chaos-text">No Matching Scenarios</h3>
          <p className="text-chaos-muted mb-4">No scenarios match your current filters.</p>
          <button
            onClick={() => setComplexityFilter(null)}
            className="text-sm text-chaos-green hover:underline"
          >
            Clear filters
          </button>
        </div>
      )}

      {/* Grid of Scenarios */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        {filteredScenarios.map((tool) => {
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
