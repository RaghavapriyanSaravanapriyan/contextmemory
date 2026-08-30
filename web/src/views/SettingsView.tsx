import React, { useState } from "react";
import { appStore, useStore, sel } from "../lib/store";

export const SettingsView: React.FC = () => {
  const settingsOpen = useStore(appStore, sel((s) => s.settingsOpen));
  const currentUrl = useStore(appStore, sel((s) => s.serverUrl));
  const currentApiKey = useStore(appStore, sel((s) => s.apiKey));
  const currentOllamaUrl = useStore(appStore, sel((s) => s.ollamaEndpoint));
  const currentModel = useStore(appStore, sel((s) => s.activeModel));
  const availableModels = useStore(appStore, sel((s) => s.availableModels));

  const [serverUrl, setServerUrl] = useState(currentUrl);
  const [apiKey, setApiKey] = useState(currentApiKey);
  const [ollamaUrl, setOllamaUrl] = useState(currentOllamaUrl);
  const [model, setModel] = useState(currentModel);

  if (!settingsOpen) return null;

  const handleClose = () => {
    appStore.setState({ settingsOpen: false });
  };

  const handleSave = () => {
    localStorage.setItem("cm_url", serverUrl);
    localStorage.setItem("cm_key", apiKey);
    localStorage.setItem("cm_ollama_url", ollamaUrl);
    localStorage.setItem("cm_model", model);

    appStore.setState({
      serverUrl,
      apiKey,
      ollamaEndpoint: ollamaUrl,
      activeModel: model,
      settingsOpen: false,
    });
  };

  const handleRestartOnboarding = () => {
    localStorage.removeItem("cm_onboarded");
    appStore.setState({ isOnboarded: false, settingsOpen: false });
  };

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div className="glass-panel modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">System Settings</div>
          <button className="glass-btn icon-only" style={{ width: "30px", height: "30px" }} onClick={handleClose}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="modal-body">
          <div className="form-group">
            <label className="form-label">ContextMemory Server URL</label>
            <input
              type="text"
              className="glass-input"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">API Key</label>
            <input
              type="password"
              className="glass-input"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Bearer secret key"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Ollama Endpoint</label>
            <input
              type="text"
              className="glass-input"
              value={ollamaUrl}
              onChange={(e) => setOllamaUrl(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Active Ollama Model</label>
            <select
              className="glass-input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{ background: "rgba(0,0,0,0.6)" }}
            >
              {availableModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", gap: "12px", marginTop: "10px" }}>
            <button className="glass-btn" onClick={handleRestartOnboarding} style={{ flex: 1 }}>
              Re-run Onboarding
            </button>
            <button className="glass-btn primary" onClick={handleSave} style={{ flex: 1 }}>
              Save Configuration
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
