"use client";

import {
  Cpu, Globe, Server, Activity, Zap, Play, Download,
  Network, Database, CheckCircle2, Shield, HardDrive,
  ChevronDown, AlertTriangle, Target, Eye
} from "lucide-react";
import { useState, useEffect, useRef, useMemo } from "react";
import { useRouter } from "next/navigation";

// ── Types ──────────────────────────────────────────────────────────────
type FaultNode = {
  name: string;
  description: string;
  apply_fn: string;
  params: Record<string, any>;
};

type CascadeNode = {
  condition_fn: string;
  condition_params: Record<string, any>;
  effect: FaultNode;
};

type ObjectiveNode = {
  description: string;
  check_fn: string;
  check_params: Record<string, any>;
  points: number;
};

type ScenarioDetail = {
  key: string;
  name: string;
  difficulty: string;
  description: string;
  max_steps: number;
  hints: string[];
  faults: FaultNode[];
  cascades: CascadeNode[];
  objectives: ObjectiveNode[];
};

type ScenarioSummary = {
  name: string;
  difficulty: string;
  description: string;
  objectives_count: number;
  max_steps: number;
};

// ── Icon helper ────────────────────────────────────────────────────────
function faultIcon(applyFn: string, className = "w-4 h-4") {
  const map: Record<string, React.ReactNode> = {
    crash_service: <Server className={className} />,
    fill_disk: <HardDrive className={className} />,
    corrupt_config: <AlertTriangle className={className} />,
    bad_permissions: <Shield className={className} />,
    add_log_flood: <Database className={className} />,
    kill_port: <Globe className={className} />,
    memory_pressure: <Activity className={className} />,
    add_unauthorized_access: <Shield className={className} />,
    write_file: <HardDrive className={className} />,
    noop: <Eye className={className} />,
    fail_cron: <Cpu className={className} />,
    drop_cron: <Cpu className={className} />,
  };
  return map[applyFn] || <Zap className={className} />;
}

function difficultyBadge(diff: string) {
  const map: Record<string, { label: string; cls: string }> = {
    easy: { label: "SAFE", cls: "text-chaos-green bg-chaos-green/10 border-chaos-green/30" },
    medium: { label: "MEDIUM", cls: "text-chaos-cyan bg-chaos-cyan/10 border-chaos-cyan/30" },
    hard: { label: "CRITICAL", cls: "text-chaos-red bg-chaos-red/10 border-chaos-red/30" },
    expert: { label: "EXPERT", cls: "text-chaos-red bg-chaos-red/20 border-chaos-red/50" },
  };
  return map[diff] || { label: diff.toUpperCase(), cls: "text-chaos-muted bg-chaos-darker border-chaos-border" };
}

// ── Builder Page ───────────────────────────────────────────────────────
export default function BuilderPage() {
  const router = useRouter();
  const canvasRef = useRef<HTMLDivElement>(null);

  const [scenarios, setScenarios] = useState<Record<string, ScenarioSummary>>({});
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [detail, setDetail] = useState<ScenarioDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectorOpen, setSelectorOpen] = useState(false);

  // Fetch scenario list on mount
  useEffect(() => {
    fetch("/api/v1/scenarios")
      .then(r => r.json())
      .then(data => {
        const s = data.scenarios || {};
        setScenarios(s);
        const keys = Object.keys(s);
        if (keys.length > 0) {
          // Default to a visually interesting one
          const defaultKey = keys.includes("cascading_db_failure") ? "cascading_db_failure" : keys[0];
          setSelectedKey(defaultKey);
        }
      })
      .catch(console.error);
  }, []);

  // Fetch full details when selection changes
  useEffect(() => {
    if (!selectedKey) return;
    setLoading(true);
    fetch(`/api/v1/scenarios/${selectedKey}`)
      .then(r => r.json())
      .then(data => {
        // Normalize: ensure arrays always exist
        setDetail({
          ...data,
          faults: data.faults || [],
          cascades: data.cascades || [],
          objectives: data.objectives || [],
          hints: data.hints || [],
          key: data.key || selectedKey,
        });
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [selectedKey]);

  // Generate YAML config from detail
  const yamlConfig = useMemo(() => {
    if (!detail) return "";
    const faults = detail.faults || [];
    const cascades = detail.cascades || [];
    const objectives = detail.objectives || [];
    const lines: string[] = [
      `scenario: "${detail.name}"`,
      `key: "${detail.key}"`,
      `difficulty: ${detail.difficulty}`,
      `max_steps: ${detail.max_steps}`,
      "",
      "faults:",
    ];
    faults.forEach((f, i) => {
      lines.push(`  - id: "fault-${i}"`);
      lines.push(`    name: "${f.name}"`);
      lines.push(`    type: "${f.apply_fn}"`);
      if (Object.keys(f.params).length > 0) {
        lines.push(`    args: ${JSON.stringify(f.params)}`);
      }
    });
    if (cascades.length > 0) {
      lines.push("");
      lines.push("cascades:");
      cascades.forEach(c => {
        lines.push(`  - condition: "${c.condition_fn}"`);
        if (Object.keys(c.condition_params).length > 0) {
          lines.push(`    when: ${JSON.stringify(c.condition_params)}`);
        }
        lines.push(`    trigger: "${c.effect.apply_fn}"`);
        if (Object.keys(c.effect.params).length > 0) {
          lines.push(`    opts: ${JSON.stringify(c.effect.params)}`);
        }
      });
    }
    lines.push("");
    lines.push("objectives:");
    objectives.forEach(o => {
      lines.push(`  - check: "${o.check_fn}"`);
      lines.push(`    description: "${o.description}"`);
      lines.push(`    points: ${o.points}`);
    });
    return lines.join("\n");
  }, [detail]);

  const handleExecute = () => {
    if (selectedKey) {
      router.push(`/playground?scenario=${selectedKey}`);
    }
  };

  const handleExport = () => {
    if (!yamlConfig) return;
    const blob = new Blob([yamlConfig], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedKey || "scenario"}.yaml`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const badge = detail ? difficultyBadge(detail.difficulty) : null;

  return (
    <div className="flex w-full h-[calc(100vh-64px)] bg-chaos-dark overflow-hidden">

      {/* ─── Left Sidebar: Scenario Selector + Palette ─── */}
      <div className="w-[280px] bg-chaos-panel/30 border-r border-chaos-border flex flex-col shrink-0">

        {/* Scenario Selector */}
        <div className="p-4 border-b border-chaos-border">
          <div className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-3">Select Scenario</div>
          <div className="relative">
            <button
              onClick={() => setSelectorOpen(!selectorOpen)}
              className="w-full flex items-center justify-between bg-chaos-panel border border-chaos-border rounded px-3 py-2.5 text-sm font-bold hover:border-chaos-green/50 transition-colors text-left"
            >
              <span className="truncate">{detail?.name || "Loading..."}</span>
              <ChevronDown className={`w-4 h-4 text-chaos-muted shrink-0 transition-transform ${selectorOpen ? 'rotate-180' : ''}`} />
            </button>

            {selectorOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-chaos-panel border border-chaos-border rounded shadow-xl z-50 max-h-[300px] overflow-y-auto">
                {Object.entries(scenarios).map(([key, s]) => {
                  const b = difficultyBadge(s.difficulty);
                  return (
                    <button
                      key={key}
                      onClick={() => { setSelectedKey(key); setSelectorOpen(false); }}
                      className={`w-full text-left px-3 py-2.5 text-sm hover:bg-chaos-darker transition-colors flex items-center justify-between gap-2 ${key === selectedKey ? 'bg-chaos-darker' : ''}`}
                    >
                      <span className="truncate font-medium">{s.name}</span>
                      <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border shrink-0 ${b.cls}`}>{b.label}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Fault Palette (from current scenario) */}
        <div className="p-4 border-b border-chaos-border flex-1 overflow-y-auto">
          <div className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-3">
            Injected Faults <span className="text-chaos-text">{detail?.faults.length || 0}</span>
          </div>
          <div className="space-y-2">
            {detail?.faults.map((f, i) => (
              <div key={i} className="flex items-center gap-3 p-3 bg-chaos-panel border border-chaos-border rounded border-l-2 border-l-chaos-red/70 transition-colors group">
                <div className="text-chaos-red">{faultIcon(f.apply_fn)}</div>
                <div className="min-w-0">
                  <div className="text-sm font-bold truncate group-hover:text-chaos-red transition-colors">{f.name}</div>
                  <div className="text-[10px] text-chaos-muted uppercase truncate">{f.apply_fn.replace(/_/g, " ")}</div>
                </div>
              </div>
            ))}
            {detail?.faults.length === 0 && (
              <div className="text-xs text-chaos-muted italic p-2">No injectable faults</div>
            )}
          </div>
        </div>

        {/* Cascade Rules */}
        <div className="p-4 border-b border-chaos-border">
          <div className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-3">
            Cascade Rules <span className="text-chaos-text">{detail?.cascades.length || 0}</span>
          </div>
          <div className="space-y-2">
            {detail?.cascades.map((c, i) => (
              <div key={i} className="flex items-center gap-3 p-3 bg-chaos-panel border border-chaos-border rounded border-l-2 border-l-chaos-cyan/70 transition-colors group">
                <Network className="w-4 h-4 text-chaos-cyan" />
                <div className="min-w-0">
                  <div className="text-sm font-bold truncate group-hover:text-chaos-cyan transition-colors">{c.condition_fn.replace(/_/g, " ")}</div>
                  <div className="text-[10px] text-chaos-muted uppercase truncate">→ {c.effect.name}</div>
                </div>
              </div>
            ))}
            {(!detail?.cascades || detail.cascades.length === 0) && (
              <div className="text-xs text-chaos-muted italic p-2">No cascade rules</div>
            )}
          </div>
        </div>

        {/* Objectives Reference */}
        <div className="p-4">
          <div className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-3">
            Objectives <span className="text-chaos-text">{detail?.objectives.length || 0}</span>
          </div>
          <div className="space-y-2">
            {detail?.objectives.map((o, i) => (
              <div key={i} className="flex items-center gap-3 p-2.5 bg-chaos-panel border border-chaos-border rounded border-l-2 border-l-chaos-green/70 transition-colors">
                <Target className="w-3.5 h-3.5 text-chaos-green shrink-0" />
                <div className="text-[11px] text-chaos-muted truncate">{o.description}</div>
                <span className="text-[10px] font-mono text-chaos-green shrink-0">{Math.round(o.points * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ─── Main Builder Canvas ─── */}
      <div className="flex-1 relative overflow-auto" ref={canvasRef}>
        {/* Grid background */}
        <div className="absolute inset-0"
          style={{
            backgroundImage: `radial-gradient(circle, rgba(57,255,20,0.03) 1px, transparent 1px)`,
            backgroundSize: '24px 24px',
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-br from-chaos-dark via-chaos-dark to-chaos-darker/50" />

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-30">
            <div className="text-chaos-green animate-pulse font-mono text-lg">Loading Scenario Graph...</div>
          </div>
        )}

        {detail && !loading && (
          <>
            {/* Canvas Header */}
            <div className="absolute top-0 left-0 right-0 p-4 flex justify-around text-[10px] font-bold text-chaos-muted uppercase tracking-widest z-10 pointer-events-none">
              <div>Fault Nodes <span className="ml-2 text-chaos-text">{detail.faults.length} Active</span></div>
              {detail.cascades.length > 0 && (
                <div>Cascade Logic <span className="ml-2 text-chaos-text">{detail.cascades.length} Trigger{detail.cascades.length > 1 ? 's' : ''}</span></div>
              )}
              <div>Validation Targets <span className="ml-2 text-chaos-text">{detail.objectives.length} Criteria</span></div>
            </div>

            {/* Node Graph */}
            <div className="relative z-10 p-12 min-w-[800px] min-h-[500px] flex items-start justify-around pt-20 gap-8">

              {/* ── Column 1: Fault Nodes ── */}
              <div className="flex flex-col gap-6 relative min-w-[240px]">
                {detail.faults.map((f, i) => (
                  <div key={i} className="w-[240px] bg-chaos-panel border border-chaos-red/50 rounded-lg p-4 shadow-[0_0_15px_rgba(255,51,51,0.1)] hover:border-chaos-red/80 transition-all hover:shadow-[0_0_25px_rgba(255,51,51,0.2)] group relative">
                    <div className="flex justify-between items-center mb-3">
                      <div className="flex flex-col min-w-0">
                        <span className="text-sm font-bold text-chaos-text truncate">{f.name.toUpperCase().replace(/ /g, "_")}</span>
                        <span className="text-[10px] font-mono text-chaos-red">ID: {f.apply_fn.substring(0, 3).toUpperCase()}-{String(i).padStart(2, '0')}</span>
                      </div>
                      <div className="text-chaos-red">{faultIcon(f.apply_fn)}</div>
                    </div>
                    <div className="text-[11px] text-chaos-muted mb-3 line-clamp-2">{f.description}</div>
                    {Object.keys(f.params).length > 0 && (
                      <div className="space-y-1.5">
                        {Object.entries(f.params).map(([k, v]) => (
                          <div key={k} className="flex justify-between text-xs">
                            <span className="text-chaos-muted capitalize">{k.replace(/_/g, " ")}</span>
                            <span className="text-chaos-green font-mono truncate max-w-[120px]">{typeof v === "number" && v > 100 ? `${(v / 1000).toFixed(1)}K` : String(v).length > 24 ? String(v).substring(0, 24) + "…" : String(v)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Connect dot */}
                    <div className="absolute right-0 top-1/2 translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-chaos-red border-[3px] border-chaos-panel z-20" />
                  </div>
                ))}
                {detail.faults.length === 0 && (
                  <div className="w-[240px] bg-chaos-panel/50 border border-dashed border-chaos-border rounded-lg p-6 text-center">
                    <Eye className="w-6 h-6 text-chaos-muted mx-auto mb-2" />
                    <div className="text-xs text-chaos-muted">Observation Only</div>
                  </div>
                )}
              </div>

              {/* ── SVG Connections ── */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
                {/* Fault → Cascade connections */}
                {detail.cascades.length > 0 && detail.faults.length > 0 && (
                  <>
                    <line
                      x1="34%" y1="40%"
                      x2={detail.cascades.length > 0 ? "46%" : "66%"}
                      y2="45%"
                      stroke="var(--color-chaos-cyan)"
                      strokeWidth="1.5"
                      strokeDasharray="6,4"
                      className="animate-pulse"
                      opacity="0.6"
                    />
                    <line
                      x1="60%" y1="45%"
                      x2="70%" y2="40%"
                      stroke="var(--color-chaos-green)"
                      strokeWidth="1.5"
                      strokeDasharray="6,4"
                      className="animate-pulse"
                      opacity="0.6"
                    />
                  </>
                )}
                {/* Direct Fault → Objective connections (no cascades) */}
                {detail.cascades.length === 0 && detail.faults.length > 0 && (
                  <line
                    x1="38%" y1="40%"
                    x2="62%" y2="40%"
                    stroke="var(--color-chaos-green)"
                    strokeWidth="1.5"
                    strokeDasharray="6,4"
                    className="animate-pulse"
                    opacity="0.6"
                  />
                )}
              </svg>

              {/* ── Column 2: Cascade Logic ── */}
              {detail.cascades.length > 0 && (
                <div className="flex flex-col gap-6 relative min-w-[280px]">
                  {detail.cascades.map((c, i) => (
                    <div key={i} className="relative">
                      {/* Connect dot left */}
                      <div className="absolute left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-chaos-cyan border-[3px] border-chaos-panel z-20" />

                      <div className="w-[280px] bg-chaos-panel border border-chaos-cyan/50 rounded-lg overflow-hidden shadow-[0_0_20px_rgba(0,255,255,0.08)] hover:shadow-[0_0_30px_rgba(0,255,255,0.15)] transition-all">
                        <div className="bg-chaos-darker border-b border-chaos-border p-3 flex justify-between items-center">
                          <span className="font-bold flex items-center gap-2 text-sm">
                            <Network className="w-4 h-4 text-chaos-cyan" /> TRIGGER CONDITION
                          </span>
                          <span className="text-[10px] font-mono text-chaos-cyan">#{i + 1}</span>
                        </div>
                        <div className="p-3 bg-chaos-cyan/5">
                          <div className="font-mono text-xs">
                            <span className="text-chaos-cyan font-bold">IF</span>{" "}
                            <span className="text-chaos-red">{c.condition_fn.replace(/_/g, " ")}</span>
                            {Object.keys(c.condition_params).length > 0 && (
                              <span className="text-chaos-text ml-1">
                                ({Object.entries(c.condition_params).map(([k, v]) => `${k}: ${v}`).join(", ")})
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="p-3 bg-chaos-panel border-t border-chaos-border">
                          <div className="flex items-center justify-between font-mono mb-1.5">
                            <span className="text-chaos-cyan font-bold text-xs">
                              ACTION: {c.effect.apply_fn.toUpperCase().replace(/_/g, "_")}
                            </span>
                            <Zap className="w-3.5 h-3.5 text-chaos-cyan" />
                          </div>
                          <div className="text-[11px] text-chaos-muted">{c.effect.description}</div>
                          {Object.keys(c.effect.params).length > 0 && (
                            <div className="flex gap-3 text-[10px] font-mono text-chaos-muted mt-2">
                              {Object.entries(c.effect.params).map(([k, v]) => (
                                <span key={k}>{k}: {String(v)}</span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Connect dot right */}
                      <div className="absolute right-0 top-1/2 translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-chaos-green border-[3px] border-chaos-panel z-20" />
                    </div>
                  ))}
                </div>
              )}

              {/* ── Column 3: Objectives / Validation ── */}
              <div className="flex flex-col gap-6 relative min-w-[220px]">
                {detail.objectives.map((o, i) => (
                  <div key={i} className="relative">
                    {/* Connect dot left */}
                    <div className="absolute left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-chaos-green border-[3px] border-chaos-panel z-20" />

                    <div className="w-[220px] bg-chaos-panel border border-chaos-border rounded-lg p-4 hover:border-chaos-green/50 transition-all bg-chaos-green/[0.02] hover:bg-chaos-green/5">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="w-2 h-2 rounded-full bg-chaos-green animate-pulse" />
                        <span className="text-[10px] font-bold text-chaos-muted font-mono uppercase tracking-wider">
                          OBJ-{String(i + 1).padStart(2, '0')}
                        </span>
                        <span className="ml-auto text-sm font-bold text-chaos-green font-mono">{Math.round(o.points * 100)}%</span>
                      </div>
                      <div className="text-xs text-chaos-text mb-3 leading-relaxed">{o.description}</div>
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-chaos-muted font-mono bg-chaos-darker px-2 py-1 rounded border border-chaos-border">
                          {o.check_fn.replace(/_/g, " ")}
                        </span>
                        <div className="w-5 h-5 rounded-full bg-chaos-green/20 flex items-center justify-center border border-chaos-green/50">
                          <CheckCircle2 className="w-3 h-3 text-chaos-green" />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* ─── Right Sidebar: Scenario Config ─── */}
      <div className="w-[320px] bg-chaos-panel/30 border-l border-chaos-border shrink-0 flex flex-col">
        <div className="p-6 border-b border-chaos-border">
          <div className="text-[10px] font-bold text-chaos-muted uppercase tracking-widest mb-1">Active Scenario</div>
          <h2 className="text-lg font-bold mb-3">{detail?.name || "Select a scenario"}</h2>
          {badge && (
            <div className="flex items-center gap-3">
              <span className={`text-[10px] font-bold px-2 py-1 rounded border ${badge.cls}`}>{badge.label}</span>
              <span className="text-[10px] font-bold text-chaos-muted">MAX: {detail?.max_steps} STEPS</span>
            </div>
          )}
        </div>

        <div className="p-6 flex-1 overflow-y-auto">
          {/* Description */}
          {detail && (
            <div className="mb-6">
              <p className="text-xs text-chaos-muted leading-relaxed">{detail.description}</p>
            </div>
          )}

          {/* Live Projections */}
          {detail && (
            <div className="mb-8">
              <h3 className="text-xs font-bold text-chaos-muted uppercase tracking-widest mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4" /> Live Projections
              </h3>
              <div className="space-y-3 font-mono text-sm">
                <div className="flex justify-between">
                  <span>Fault Nodes</span>
                  <span className="text-chaos-red">{detail.faults.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Cascade Depth</span>
                  <span className="text-chaos-cyan">{detail.cascades.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Objectives</span>
                  <span className="text-chaos-green">{detail.objectives.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Blast Radius</span>
                  <span className="text-chaos-red">
                    {((detail.faults.length + detail.cascades.length) / (detail.objectives.length || 1) * 20).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Complexity</span>
                  <span className="text-chaos-cyan">
                    {(0.3 + detail.faults.length * 0.12 + detail.cascades.length * 0.2).toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Hints */}
          {detail && detail.hints.length > 0 && (
            <div className="mb-8">
              <h3 className="text-xs font-bold text-chaos-muted uppercase tracking-widest mb-3">Hints</h3>
              <ul className="space-y-2">
                {detail.hints.map((h, i) => (
                  <li key={i} className="text-[11px] text-chaos-muted flex gap-2">
                    <span className="text-chaos-cyan shrink-0">›</span>
                    {h}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* YAML Config Manifest */}
          <div>
            <h3 className="text-xs font-bold text-chaos-muted uppercase tracking-widest mb-4 flex items-center gap-2">
              <Database className="w-4 h-4" /> Config Manifest
            </h3>
            <pre className="text-[10px] font-mono text-chaos-green bg-chaos-darker p-4 rounded border border-chaos-border overflow-x-auto max-h-[240px] overflow-y-auto">
              {yamlConfig || "# Select a scenario to view config"}
            </pre>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="p-4 border-t border-chaos-border bg-chaos-panel space-y-3">
          <button
            onClick={handleExecute}
            disabled={!selectedKey}
            className="w-full flex items-center justify-center gap-2 bg-chaos-green text-chaos-dark font-bold px-4 py-3 rounded hover:bg-chaos-green/90 transition-colors uppercase tracking-widest text-sm shadow-[0_0_15px_rgba(57,255,20,0.2)] disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4 fill-current" /> Execute Experiment
          </button>
          <button
            onClick={handleExport}
            disabled={!yamlConfig}
            className="w-full flex items-center justify-center gap-2 bg-transparent border border-chaos-border text-chaos-text font-bold px-4 py-3 rounded hover:bg-chaos-darker transition-colors text-xs uppercase tracking-widest disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" /> Export YAML Bundle
          </button>
        </div>
      </div>

    </div>
  );
}
