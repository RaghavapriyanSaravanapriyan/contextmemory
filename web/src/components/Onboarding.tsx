import React, { useState } from "react";
import { appStore, useStore, sel } from "../lib/store";

export const Onboarding: React.FC = () => {
  const [step, setStep] = useState<number>(1);
  const [serverUrl, setServerUrl] = useState<string>("http://localhost:8765");
  const [apiKey, setApiKey] = useState<string>("");
  const [ollamaUrl, setOllamaUrl] = useState<string>("http://localhost:11434");
  const [selectedModel, setSelectedModel] = useState<string>("qwen3:4b");
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<string>("");

  const availableModels = useStore(appStore, sel((s) => s.availableModels));

  const handleTestConnection = async () => {
    setIsTesting(true);
    setTestResult("Connecting to ContextMemory server...");
    try {
      const res = await fetch(`${serverUrl}/v1/health`);
      if (res.ok) {
        setTestResult("Connected! Local engine verified.");
        setTimeout(() => setStep(3), 800);
      } else {
        setTestResult("Server not reached. Live demo mode enabled.");
        setTimeout(() => setStep(3), 800);
      }
    } catch {
      setTestResult("Server offline. Live demo mode enabled.");
      setTimeout(() => setStep(3), 800);
    } finally {
      setIsTesting(false);
    }
  };

  const handleFinish = () => {
    localStorage.setItem("cm_onboarded", "true");
    localStorage.setItem("cm_url", serverUrl);
    localStorage.setItem("cm_key", apiKey);
    localStorage.setItem("cm_ollama_url", ollamaUrl);
    localStorage.setItem("cm_model", selectedModel);

    appStore.setState({
      isOnboarded: true,
      serverUrl,
      apiKey,
      ollamaEndpoint: ollamaUrl,
      activeModel: selectedModel,
    });
  };

  return (
    <div className="onboarding-screen">
      <div className="glass-panel onboarding-card">
        <div className="onboarding-steps">
          {[1, 2, 3, 4].map((s) => (
            <div key={s} className={`step-dot ${s === step ? "active" : ""}`} />
          ))}
        </div>

        {step === 1 && (
          <>
            <h1 className="onboarding-title">Your AI remembers.</h1>
            <p className="onboarding-subtitle">
              ContextMemory provides local LLMs with high-throughput persistent graph memory,
              deterministic retrieval, and real-time synaptic visibility.
            </p>
            <button className="glass-btn primary" onClick={() => setStep(2)}>
              Initialize Memory System →
            </button>
          </>
        )}

        {step === 2 && (
          <>
            <h2 className="onboarding-title" style={{ fontSize: "24px" }}>
              Connect ContextMemory
            </h2>
            <p className="onboarding-subtitle">
              Configure your local ContextMemory FastAPI server instance.
            </p>

            <div className="onboarding-form">
              <div className="form-group">
                <label className="form-label">Server Endpoint</label>
                <input
                  type="text"
                  className="glass-input"
                  value={serverUrl}
                  onChange={(e) => setServerUrl(e.target.value)}
                  placeholder="http://localhost:8765"
                />
              </div>

              <div className="form-group">
                <label className="form-label">API Key (Optional)</label>
                <input
                  type="password"
                  className="glass-input"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Bearer token or leave blank"
                />
              </div>
            </div>

            {testResult && (
              <div style={{ fontSize: "12px", color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
                {testResult}
              </div>
            )}

            <div style={{ display: "flex", gap: "12px", width: "100%" }}>
              <button
                className="glass-btn"
                style={{ flex: 1 }}
                onClick={() => setStep(3)}
              >
                Skip / Offline Demo
              </button>
              <button
                className="glass-btn primary"
                style={{ flex: 1 }}
                onClick={handleTestConnection}
                disabled={isTesting}
              >
                {isTesting ? "Testing..." : "Connect & Continue →"}
              </button>
            </div>
          </>
        )}

        {step === 3 && (
          <>
            <h2 className="onboarding-title" style={{ fontSize: "24px" }}>
              Local Model Selection
            </h2>
            <p className="onboarding-subtitle">
              Select an Ollama model for question answering and context extraction.
            </p>

            <div className="onboarding-form">
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
                <label className="form-label">Select Active Model</label>
                <select
                  className="glass-input"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  style={{ background: "rgba(0,0,0,0.6)" }}
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button className="glass-btn primary" onClick={() => setStep(4)} style={{ width: "100%" }}>
              Confirm Model →
            </button>
          </>
        )}

        {step === 4 && (
          <>
            <div className="brand-icon" style={{ width: "42px", height: "42px" }} />
            <h1 className="onboarding-title" style={{ fontSize: "28px" }}>
              Memory System Online
            </h1>
            <p className="onboarding-subtitle">
              Neural graph indexing ready. All memories are stored locally and encrypted.
            </p>
            <button className="glass-btn primary" onClick={handleFinish} style={{ width: "100%", padding: "12px" }}>
              Enter Brain Observatory
            </button>
          </>
        )}
      </div>
    </div>
  );
};
