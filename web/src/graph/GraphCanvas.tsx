import React, { useEffect, useRef } from "react";
import { BrainGraph } from "./BrainGraph";
import { appStore, useStore, sel } from "../lib/store";
import type { GraphNode } from "../lib/types";

export const GraphCanvas: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const brainRef = useRef<BrainGraph | null>(null);

  const graphData = useStore(appStore, sel((s) => s.graphData));
  const activeTrace = useStore(appStore, sel((s) => s.activeTrace));
  const selectedNodeId = useStore(appStore, sel((s) => s.selectedNodeId));

  useEffect(() => {
    if (!containerRef.current) return;

    const brain = new BrainGraph(containerRef.current);
    brainRef.current = brain;

    brain.onNodeSelect = (node: GraphNode | null) => {
      appStore.setState({
        selectedNodeId: node ? node.id : null,
        selectedNode: node,
      });
    };

    return () => {
      brain.destroy();
      brainRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (brainRef.current && graphData) {
      brainRef.current.setData(graphData);
    }
  }, [graphData]);

  useEffect(() => {
    if (brainRef.current && activeTrace && activeTrace.evidence) {
      brainRef.current.triggerSynapticFlow(activeTrace.evidence);
    }
  }, [activeTrace]);

  useEffect(() => {
    if (brainRef.current) {
      if (selectedNodeId) {
        brainRef.current.focusNode(selectedNodeId);
      } else {
        brainRef.current.resetCamera();
      }
    }
  }, [selectedNodeId]);

  return <div ref={containerRef} className="canvas-container" />;
};
