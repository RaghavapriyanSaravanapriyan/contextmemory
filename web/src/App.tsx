import { useEffect } from "react";
import { appStore, useStore, sel } from "./lib/store";
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
