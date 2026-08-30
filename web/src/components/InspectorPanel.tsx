import React from "react";
import { appStore, useStore, sel } from "../lib/store";
import { api } from "../lib/api";

export const InspectorPanel: React.FC = () => {
  const selectedNode = useStore(appStore, sel((s) => s.selectedNode));
  const graphData = useStore(appStore, sel((s) => s.graphData));

  if (!selectedNode) return null;

  const handleClose = () => {
    appStore.setState({ selectedNodeId: null, selectedNode: null });
  };

  const handleForget = async () => {
    if (!selectedNode) return;
    try {
      await api.forget(selectedNode.space_id, selectedNode.id);
    } catch {
      appStore.setState((prev) => ({
        graphData: {
          ...prev.graphData,
          nodes: prev.graphData.nodes.filter((n) => n.id !== selectedNode.id),
          edges: prev.graphData.edges.filter(
            (e) => e.from !== selectedNode.id && e.to !== selectedNode.id
          ),
        },
        selectedNodeId: null,
        selectedNode: null,
      }));
    }
  };

  const connectedEdges = graphData.edges.filter(
    (e) => e.from === selectedNode.id || e.to === selectedNode.id
  );
  const connectedNodeIds = new Set(
    connectedEdges.map((e) => (e.from === selectedNode.id ? e.to : e.from))
  );
  const connectedNodes = graphData.nodes.filter((n) => connectedNodeIds.has(n.id));

  return (
    <div className="glass-panel inspector-drawer">
      <div className="inspector-header">
        <div className="inspector-title-group">
          <span className="node-kind-badge">{selectedNode.kind}</span>
          <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
            ID: {selectedNode.id.substring(0, 10)}...
          </span>
        </div>

        <button className="glass-btn icon-only" style={{ width: "28px", height: "28px" }} onClick={handleClose}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div className="inspector-body">
        <div className="memory-text-card">{selectedNode.text}</div>

        <div className="triple-box">
          <div className="triple-cell">
            <span className="triple-label">Subject</span>
            <span className="triple-val">{selectedNode.subject || "—"}</span>
          </div>
          <div className="triple-cell">
            <span className="triple-label">Predicate</span>
            <span className="triple-val">{selectedNode.predicate || "—"}</span>
          </div>
          <div className="triple-cell">
            <span className="triple-label">Object</span>
            <span className="triple-val">{selectedNode.object || "—"}</span>
          </div>
        </div>

        <div className="metrics-bar-group">
          <div className="metric-progress">
            <div className="metric-progress-label">
              <span>Confidence</span>
              <span>{Math.round((selectedNode.confidence || 0.9) * 100)}%</span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${(selectedNode.confidence || 0.9) * 100}%` }}
              />
            </div>
          </div>

          <div className="metric-progress">
            <div className="metric-progress-label">
              <span>Salience / Importance</span>
              <span>{Math.round((selectedNode.salience || 0.8) * 100)}%</span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${(selectedNode.salience || 0.8) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {selectedNode.tags && selectedNode.tags.length > 0 && (
          <div>
            <div className="form-label" style={{ marginBottom: "8px" }}>
              Tags
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {selectedNode.tags.map((t) => (
                <span
                  key={t}
                  style={{
                    fontSize: "10px",
                    fontFamily: "var(--font-mono)",
                    color: "var(--text-secondary)",
                    background: "rgba(255,255,255,0.06)",
                    padding: "3px 8px",
                    borderRadius: "4px",
                    border: "var(--border-glass)",
                  }}
                >
                  #{t}
                </span>
              ))}
            </div>
          </div>
        )}

        {connectedNodes.length > 0 && (
          <div>
            <div className="form-label" style={{ marginBottom: "8px" }}>
              Connected Synapses ({connectedNodes.length})
            </div>
            <div className="evidence-list">
              {connectedNodes.map((cn) => (
                <div
                  key={cn.id}
                  className="evidence-item"
                  onClick={() => appStore.setState({ selectedNodeId: cn.id, selectedNode: cn })}
                >
                  <div style={{ fontSize: "12px", color: "var(--text-pure)" }}>{cn.subject || cn.text}</div>
                  <div style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    {cn.kind} • salience: {Math.round((cn.salience || 0.5) * 100)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          className="glass-btn"
          style={{ color: "var(--status-err)", borderColor: "rgba(255, 69, 58, 0.3)", marginTop: "auto" }}
          onClick={handleForget}
        >
          Forget Memory Node
        </button>
      </div>
    </div>
  );
};
