import React from "react";
import { appStore, useStore, sel } from "../lib/store";

export const TopBar: React.FC = () => {
  const connection = useStore(appStore, sel((s) => s.connection));
  const activeModel = useStore(appStore, sel((s) => s.activeModel));
  const activeSpaceId = useStore(appStore, sel((s) => s.activeSpaceId));

  const handleResetCamera = () => {
    appStore.setState({ selectedNodeId: null, selectedNode: null });
  };

  return (
    <header className="topbar">
      <div className="brand-group">
        <div className="brand-icon" />
        <div className="brand-title">
          ContextMemory
          <span className="brand-tag">v2.0</span>
        </div>
      </div>

      <div className="topbar-center">
        <div className={`status-pill ${connection.phase === "live" ? "online" : ""}`}>
          <div className="status-dot" />
          <span>{connection.phase === "live" ? "ENGINE ONLINE" : "OFFLINE DEMO"}</span>
        </div>

        <div className="model-badge" onClick={() => appStore.setState({ settingsOpen: true })}>
          <span style={{ color: "var(--text-muted)", fontSize: "10px" }}>MODEL</span>
          <span>{activeModel}</span>
        </div>

        <div className="status-pill">
          <span style={{ color: "var(--text-muted)" }}>SPACE</span>
          <span style={{ color: "var(--text-pure)" }}>{activeSpaceId}</span>
        </div>
      </div>

      <div className="topbar-right">
        <button
          className="glass-btn icon-only"
          title="Reset Camera View"
          onClick={handleResetCamera}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
          </svg>
        </button>

        <button
          className="glass-btn"
          onClick={() => appStore.setState({ historyModalOpen: true })}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          History
        </button>

        <button
          className="glass-btn icon-only"
          title="Settings"
          onClick={() => appStore.setState({ settingsOpen: true })}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>
    </header>
  );
};
