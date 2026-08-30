import React from "react";
import { appStore, useStore, sel } from "../lib/store";

export const RetrievalTracePanel: React.FC = () => {
  const activeTrace = useStore(appStore, sel((s) => s.activeTrace));

  if (!activeTrace) return null;

  const handleClose = () => {
    appStore.setState({ activeTrace: null });
  };

  const handleReplay = () => {
    // Re-assign active trace to re-trigger 3D signal traversal animation
    appStore.setState({ activeTrace: { ...activeTrace } });
  };

  return (
    <div className="glass-panel trace-panel">
      {/* Header */}
      <div className="inspector-header">
        <div className="inspector-title-group">
          <span className="node-kind-badge" style={{ background: "rgba(255,255,255,0.15)" }}>
            RETRIEVAL TRACE
          </span>
          <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
            {activeTrace.tokens} tokens
          </span>
        </div>

        <button className="glass-btn icon-only" style={{ width: "28px", height: "28px" }} onClick={handleClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="trace-body">
        {/* Query */}
        <div>
          <div className="form-label" style={{ marginBottom: "6px" }}>
            Query
          </div>
          <div className="query-display">"{activeTrace.query}"</div>
        </div>

        {/* Answer */}
        {activeTrace.answer && (
          <div>
            <div className="form-label" style={{ marginBottom: "6px" }}>
              Retrieved Context & Generated Response
            </div>
            <div className="answer-card">{activeTrace.answer}</div>
          </div>
        )}

        {/* Evidence Memory Nodes */}
        <div>
          <div className="form-label" style={{ marginBottom: "8px" }}>
            Retrieved Evidence Nodes ({activeTrace.evidence?.length || 0})
          </div>
          <div className="evidence-list">
            {activeTrace.evidence?.map((hit) => (
              <div
                key={hit.id}
                className="evidence-item"
                onClick={() => appStore.setState({ selectedNodeId: hit.id })}
              >
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                  <span style={{ fontSize: "12px", color: "var(--text-pure)", fontWeight: 500 }}>
                    {hit.subject || hit.kind}
                  </span>
                  <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    score: {Math.round((hit.score || 0.9) * 100)}%
                  </span>
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-secondary)" }}>{hit.text}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Replay Synaptic Flow Button */}
        <button className="glass-btn primary" onClick={handleReplay} style={{ marginTop: "10px" }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          Replay Synaptic Flow
        </button>
      </div>
    </div>
  );
};
