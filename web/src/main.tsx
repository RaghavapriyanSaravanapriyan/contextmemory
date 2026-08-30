import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles.css";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error("[observatory]", error);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ display: "grid", placeItems: "center", height: "100vh", color: "var(--text-secondary)" }}>
          <div style={{ maxWidth: 460, fontFamily: "var(--font-mono)", fontSize: 13 }}>
            <div style={{ color: "var(--status-err)", marginBottom: 8 }}>Observatory error</div>
            <div style={{ opacity: 0.7 }}>{String(this.state.error.message || this.state.error)}</div>
            <button
              className="glass-btn"
              style={{ marginTop: 16 }}
              onClick={() => location.reload()}
            >
              Reload System
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
