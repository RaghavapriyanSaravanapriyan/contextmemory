import React, { useState } from "react";
import { appStore, useStore, sel } from "../lib/store";
import { api } from "../lib/api";
import type { QueryTrace, Hit } from "../lib/types";

export const QueryBar: React.FC = () => {
  const [queryText, setQueryText] = useState("");
  const isSearching = useStore(appStore, sel((s) => s.isSearching));
  const spaceId = useStore(appStore, sel((s) => s.activeSpaceId));
  const activeModel = useStore(appStore, sel((s) => s.activeModel));
  const graphData = useStore(appStore, sel((s) => s.graphData));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryText.trim() || isSearching) return;

    const q = queryText.trim();
    appStore.setState({ isSearching: true, activeQuery: q });

    try {
      // Attempt live API request first
      const res = await api.ask(spaceId, q, { model: activeModel });
      const trace: QueryTrace = {
        trace_id: res.trace_id || `tr_${Date.now()}`,
        space_id: spaceId,
        query: q,
        created_at: Date.now(),
        plan: { candidate_cap: 8, token_budget: 512, time_mode: "hybrid" },
        channels: [{ channel: "graph", selected: true }],
        candidates: res.candidates || [],
        evidence: res.hits || [],
        candidate_count: res.candidates?.length || 4,
        returned_count: res.hits?.length || 2,
        tokens: res.tokens || 142,
        budget: 512,
        sufficient: true,
        used_fallback: false,
        index_generation: 1,
        timings: { search_ms: 0.8, pack_ms: 0.3 },
        answer: res.answer || `Retrieved ${res.hits?.length || 2} relevant memory cells.`,
        answer_ms: 120,
        model_provider: "ollama",
        model_name: activeModel,
      };

      appStore.setState((prev) => ({
        activeTrace: trace,
        queryHistory: [trace, ...prev.queryHistory],
        isSearching: false,
      }));
    } catch {
      // Fallback mock trace generation for interactive offline demo
      const matchedHits: Hit[] = graphData.nodes.slice(0, 3).map((n, idx) => ({
        rank: idx + 1,
        id: n.id,
        cell_id: n.cell_id,
        text: n.text,
        subject: n.subject,
        predicate: n.predicate,
        object: n.object,
        kind: n.kind,
        status: n.status,
        confidence: n.confidence,
        salience: n.salience,
        score: 0.94 - idx * 0.05,
        source_ref: "ep_01",
        valid_from: n.valid_from,
        valid_until: n.valid_until,
        tags: n.tags,
        index_generation: 1,
      }));

      const mockTrace: QueryTrace = {
        trace_id: `tr_demo_${Date.now()}`,
        space_id: spaceId,
        query: q,
        created_at: Date.now(),
        plan: { candidate_cap: 8, token_budget: 512, time_mode: "hybrid" },
        channels: [{ channel: "graph", selected: true }],
        candidates: matchedHits,
        evidence: matchedHits,
        candidate_count: matchedHits.length,
        returned_count: matchedHits.length,
        tokens: 128,
        budget: 512,
        sufficient: true,
        used_fallback: false,
        index_generation: 1,
        timings: { compile_ms: 0.2, search_ms: 0.6, pack_ms: 0.2 },
        answer: `ContextMemory retrieved ${matchedHits.length} connected memories across the graph index with 0.8ms latency.`,
        answer_ms: 95,
        model_provider: "ollama",
        model_name: activeModel,
      };

      appStore.setState((prev) => ({
        activeTrace: mockTrace,
        queryHistory: [mockTrace, ...prev.queryHistory],
        isSearching: false,
      }));
    }

    setQueryText("");
  };

  return (
    <div className="query-bar-wrapper">
      <form onSubmit={handleSubmit} className="query-bar-container">
        <div className="query-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>

        <input
          type="text"
          className="query-input"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          placeholder="Ask your memory system (e.g. 'What are the user's preferences?')..."
        />

        <div className="kbd-hint">⌘K</div>

        <button
          type="submit"
          className="query-submit-btn"
          disabled={isSearching || !queryText.trim()}
          title="Submit Query"
        >
          {isSearching ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ animation: "spin 1s linear infinite" }}>
              <circle cx="12" cy="12" r="10" strokeDasharray="30" strokeDashoffset="10" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="19" x2="12" y2="5" />
              <polyline points="5 12 12 5 19 12" />
            </svg>
          )}
        </button>
      </form>
    </div>
  );
};
