import { useEffect, useRef } from "react";
import { appStore, useStore, sel } from "./lib/store";
import { api } from "./lib/api";
import type { GraphData, Hit, QueryTrace } from "./lib/types";
import { GraphCanvas } from "./graph/GraphCanvas";
import { TopBar } from "./components/TopBar";
import { QueryBar } from "./components/QueryBar";
import { SystemStatsOverlay } from "./components/SystemStatsOverlay";
import { InspectorPanel } from "./components/InspectorPanel";
import { RetrievalTracePanel } from "./components/RetrievalTracePanel";
import { RetrievalHistoryModal } from "./components/RetrievalHistoryModal";
import { SettingsView } from "./views/SettingsView";
import { Onboarding } from "./components/Onboarding";

export function App() {
  const isOnboarded = useStore(appStore, sel((s) => s.isOnboarded));
  const activeSpaceId = useStore(appStore, sel((s) => s.activeSpaceId));
  const lastProcessedEvtId = useRef<string | null>(null);

  // Poll Real-Time Graph & Events from Backend
  useEffect(() => {
    let mounted = true;

    const pollGraph = async () => {
      try {
        const data = await api.get<GraphData>(`/v1/graph?space_id=${activeSpaceId}`);
        if (mounted && data && data.nodes) {
          appStore.setState({
            graphData: data,
            connection: { phase: "live", ws: "live", engine: "ok", last_error: "" },
          });
        }
      } catch {
        if (mounted) {
          appStore.setState({
            connection: { phase: "offline", ws: "offline", engine: "down", last_error: "Backend Offline" },
          });
        }
      }
    };

    const pollEvents = async () => {
      try {
        const res = await api.get<{ events: Array<{ id: string; type: string; data: any }> }>("/v1/events");
        if (mounted && res && res.events && res.events.length > 0) {
          const latest = res.events[res.events.length - 1];
          if (latest.id !== lastProcessedEvtId.current) {
            lastProcessedEvtId.current = latest.id;

            if (latest.type === "query_executed" && latest.data) {
              const hits: Hit[] = (latest.data.hits || []).map((h: any, i: number) => ({
                id: h.id || `hit_${i}`,
                cell_id: parseInt(h.id) || i + 1,
                text: h.text || "",
                subject: h.subject || "",
                predicate: h.predicate || "",
                object: h.object || "",
                score: h.score || 0.9,
                kind: h.kind || "world",
                status: "active",
                confidence: 0.95,
                salience: h.salience || 0.8,
                source_ref: "etmc_core",
                valid_from: Date.now(),
                valid_until: 0,
                tags: [],
                index_generation: 1,
              }));

              const trace: QueryTrace = {
                trace_id: latest.id,
                space_id: activeSpaceId,
                query: latest.data.query,
                created_at: Date.now(),
                plan: {},
                channels: [{ channel: "etmc_core", selected: true }],
                candidates: hits,
                evidence: hits,
                candidate_count: hits.length,
                returned_count: hits.length,
                tokens: latest.data.tokens || 128,
                budget: 512,
                sufficient: true,
                used_fallback: false,
                index_generation: 1,
                timings: { recall_ms: latest.data.latency_ms || 1.1 },
                answer: `ContextMemory retrieved ${hits.length} connected memories across the graph index with ${latest.data.latency_ms || 1.1}ms latency.`,
                answer_ms: latest.data.latency_ms || 1.1,
                model_provider: "ollama",
                model_name: "qwen3:4b",
              };

              appStore.setState((prev) => ({
                activeTrace: trace,
                queryHistory: [trace, ...prev.queryHistory.slice(0, 49)],
              }));

              // Refetch graph immediately
              pollGraph();
            } else if (latest.type === "memory_created") {
              pollGraph();
            }
          }
        }
      } catch {
        // Suppress transient poll error
      }
    };

    // Initial fetch
    pollGraph();
    pollEvents();

    const graphInterval = setInterval(pollGraph, 1500);
    const eventInterval = setInterval(pollEvents, 1000);

    return () => {
      mounted = false;
      clearInterval(graphInterval);
      clearInterval(eventInterval);
    };
  }, [activeSpaceId]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        const inputEl = document.querySelector(".query-input") as HTMLInputElement;
        inputEl?.focus();
      }
      if (e.key === "Escape") {
        appStore.setState({
          selectedNodeId: null,
          selectedNode: null,
          activeTrace: null,
          historyModalOpen: false,
          settingsOpen: false,
        });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  if (!isOnboarded) {
    return <Onboarding />;
  }

  return (
    <div className="app-shell">
      <GraphCanvas />

      <div className="ui-overlay">
        <TopBar />
        <SystemStatsOverlay />
        <QueryBar />

        <InspectorPanel />
        <RetrievalTracePanel />

        <RetrievalHistoryModal />
        <SettingsView />
      </div>
    </div>
  );
}
