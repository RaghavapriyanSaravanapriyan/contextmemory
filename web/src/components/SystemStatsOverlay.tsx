import React from "react";
import { appStore, useStore, sel } from "../lib/store";

export const SystemStatsOverlay: React.FC = () => {
  const metrics = useStore(appStore, sel((s) => s.metrics));
  const graphData = useStore(appStore, sel((s) => s.graphData));

  const totalMemories = metrics?.counts.total || graphData.nodes.length;
  const totalEdges = graphData.edges.length;
  const avgLatency = metrics?.latency.avg_ms || 0.8;

  return (
    <div className="telemetry-overlay">
      <div className="telemetry-card">
        <div className="telemetry-label">
          <span>MEMORIES</span>
          <span style={{ color: "var(--text-pure)" }}>LIVE</span>
        </div>
        <div className="telemetry-value">
          {totalMemories}
          <span className="telemetry-unit">nodes</span>
        </div>
      </div>

      <div className="telemetry-card">
        <div className="telemetry-label">
          <span>GRAPH EDGES</span>
        </div>
        <div className="telemetry-value">
          {totalEdges}
          <span className="telemetry-unit">synapses</span>
        </div>
      </div>

      <div className="telemetry-card">
        <div className="telemetry-label">
          <span>READ LATENCY</span>
        </div>
        <div className="telemetry-value">
          {avgLatency}
          <span className="telemetry-unit">ms</span>
        </div>
      </div>
    </div>
  );
};
