import React from "react";
import { appStore, useStore, sel } from "../lib/store";
import type { QueryTrace } from "../lib/types";

export const RetrievalHistoryModal: React.FC = () => {
  const historyModalOpen = useStore(appStore, sel((s) => s.historyModalOpen));
  const queryHistory = useStore(appStore, sel((s) => s.queryHistory));

  if (!historyModalOpen) return null;

  const handleClose = () => {
    appStore.setState({ historyModalOpen: false });
  };

  const handleSelectTrace = (trace: QueryTrace) => {
    appStore.setState({
      activeTrace: trace,
      historyModalOpen: false,
    });
  };

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div className="glass-panel modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">Retrieval History</div>
          <button className="glass-btn icon-only" style={{ width: "30px", height: "30px" }} onClick={handleClose}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="modal-body">
          {queryHistory.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px 0", color: "var(--text-muted)" }}>
              No query traces recorded yet. Ask a question using the command bar!
            </div>
          ) : (
            <div className="evidence-list">
              {queryHistory.map((tr) => (
                <div
                  key={tr.trace_id}
                  className="evidence-item"
                  onClick={() => handleSelectTrace(tr)}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                    <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-pure)" }}>
                      "{tr.query}"
                    </span>
                    <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                      {new Date(tr.created_at).toLocaleTimeString()}
                    </span>
                  </div>

                  <div style={{ display: "flex", gap: "12px", fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                    <span>{tr.evidence?.length || 0} evidence nodes</span>
                    <span>{tr.tokens} tokens</span>
                    <span>model: {tr.model_name}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
