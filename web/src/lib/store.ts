import { useCallback, useRef, useSyncExternalStore } from "react";
import type { GraphData, GraphNode, QueryTrace, MetricSnapshot, ConnectionState } from "./types";

export interface AppState {
  isOnboarded: boolean;
  connection: ConnectionState;
  serverUrl: string;
  apiKey: string;
  ollamaEndpoint: string;
  activeModel: string;
  availableModels: string[];
  activeSpaceId: string;
  graphData: GraphData;
  selectedNodeId: string | null;
  selectedNode: GraphNode | null;
  activeQuery: string;
  activeTrace: QueryTrace | null;
  isSearching: boolean;
  queryHistory: QueryTrace[];
  metrics: MetricSnapshot | null;
  settingsOpen: boolean;
  historyModalOpen: boolean;
}

export const MOCK_GRAPH: GraphData = {
  space_id: "brain",
  generation: 1,
  node_count: 8,
  edge_count: 7,
  nodes: [
    {
      id: "mem_01",
      space_id: "brain",
      cell_id: 1,
      text: "User prefers concise Python code with strict type annotations and docstrings.",
      subject: "User",
      predicate: "prefers",
      object: "concise Python code",
      kind: "preference",
      status: "active",
      confidence: 0.95,
      salience: 0.92,
      valid_from: Date.now() - 86400000 * 5,
      valid_until: 0,
      observed_at: Date.now() - 86400000 * 5,
      root_id: "mem_01",
      parent_id: "",
      tags: ["coding", "python", "preference"],
      entities: ["User", "Python"],
      created_at: Date.now() - 86400000 * 5,
      updated_at: Date.now() - 86400000 * 5,
    },
    {
      id: "mem_02",
      space_id: "brain",
      cell_id: 2,
      text: "ContextMemory engine combines C++ graph indexing with SQLite temporal persistence.",
      subject: "ContextMemory",
      predicate: "uses",
      object: "C++ and SQLite",
      kind: "world",
      status: "active",
      confidence: 0.99,
      salience: 0.95,
      valid_from: Date.now() - 86400000 * 10,
      valid_until: 0,
      observed_at: Date.now() - 86400000 * 10,
      root_id: "mem_02",
      parent_id: "",
      tags: ["architecture", "cpp", "sqlite"],
      entities: ["ContextMemory", "C++", "SQLite"],
      created_at: Date.now() - 86400000 * 10,
      updated_at: Date.now() - 86400000 * 10,
    },
    {
      id: "mem_03",
      space_id: "brain",
      cell_id: 3,
      text: "Ollama local inference server operates on port 11434 with models like Qwen3:4B.",
      subject: "Ollama",
      predicate: "runs on",
      object: "http://localhost:11434",
      kind: "world",
      status: "active",
      confidence: 0.98,
      salience: 0.88,
      valid_from: Date.now() - 86400000 * 4,
      valid_until: 0,
      observed_at: Date.now() - 86400000 * 4,
      root_id: "mem_03",
      parent_id: "",
      tags: ["ollama", "llm", "local"],
      entities: ["Ollama", "Qwen3"],
      created_at: Date.now() - 86400000 * 4,
      updated_at: Date.now() - 86400000 * 4,
    },
    {
      id: "mem_04",
      space_id: "brain",
      cell_id: 4,
      text: "MCP server enables local agentic tools like memory_search and session_ingest.",
      subject: "MCP Server",
      predicate: "exposes",
      object: "memory tools",
      kind: "procedure",
      status: "active",
      confidence: 0.96,
      salience: 0.9,
      valid_from: Date.now() - 86400000 * 3,
      valid_until: 0,
      observed_at: Date.now() - 86400000 * 3,
      root_id: "mem_04",
      parent_id: "",
      tags: ["mcp", "tools", "agent"],
      entities: ["MCP Server", "ContextMemory"],
      created_at: Date.now() - 86400000 * 3,
      updated_at: Date.now() - 86400000 * 3,
    },
    {
      id: "mem_05",
      space_id: "brain",
      cell_id: 5,
      text: "User prefers monochrome dark interfaces with liquid glass translucent overlays.",
      subject: "User",
      predicate: "prefers",
      object: "Liquid Glass Dark Design",
      kind: "preference",
      status: "active",
      confidence: 0.97,
      salience: 0.96,
      valid_from: Date.now() - 86400000 * 2,
      valid_until: 0,
      observed_at: Date.now() - 86400000 * 2,
      root_id: "mem_05",
      parent_id: "",
      tags: ["ui", "design", "preference"],
      entities: ["User", "Liquid Glass"],
      created_at: Date.now() - 86400000 * 2,
      updated_at: Date.now() - 86400000 * 2,
    },
    {
      id: "mem_06",
      space_id: "brain",
      cell_id: 6,
      text: "Retrieval path generates synaptic energy signal traces across connected candidate nodes.",
      subject: "Retrieval",
      predicate: "visualizes as",
      object: "Synaptic Signal Flow",
      kind: "opinion",
      status: "active",
      confidence: 0.94,
      salience: 0.91,
      valid_from: Date.now() - 86400000 * 1,
      valid_until: 0,
      observed_at: Date.now() - 86400000 * 1,
      root_id: "mem_06",
      parent_id: "",
      tags: ["retrieval", "synapse", "graph"],
      entities: ["Retrieval", "ContextMemory"],
      created_at: Date.now() - 86400000 * 1,
      updated_at: Date.now() - 86400000 * 1,
    },
    {
      id: "mem_07",
      space_id: "brain",
      cell_id: 7,
      text: "Benchmark test confirmed 0.8ms average retrieval latency over 10,000 memory cells.",
      subject: "Benchmark",
      predicate: "measured latency",
      object: "0.8ms",
      kind: "experience",
      status: "active",
      confidence: 0.99,
      salience: 0.89,
      valid_from: Date.now() - 43200000,
      valid_until: 0,
      observed_at: Date.now() - 43200000,
      root_id: "mem_07",
      parent_id: "",
      tags: ["performance", "benchmark", "latency"],
      entities: ["Benchmark", "ContextMemory"],
      created_at: Date.now() - 43200000,
      updated_at: Date.now() - 43200000,
    },
    {
      id: "mem_08",
      space_id: "brain",
      cell_id: 8,
      text: "FastAPI server streams live events via Server-Sent Events at /v1/stream and WebSockets at /ws/events.",
      subject: "FastAPI",
      predicate: "streams",
      object: "Real-time activity events",
      kind: "procedure",
      status: "active",
      confidence: 0.98,
      salience: 0.87,
      valid_from: Date.now() - 21600000,
      valid_until: 0,
      observed_at: Date.now() - 21600000,
      root_id: "mem_08",
      parent_id: "",
      tags: ["fastapi", "sse", "websocket"],
      entities: ["FastAPI", "WebSockets"],
      created_at: Date.now() - 21600000,
      updated_at: Date.now() - 21600000,
    },
  ],
  edges: [
    { id: "e1", from: "mem_01", to: "mem_05", type: "subject", derived: true },
    { id: "e2", from: "mem_02", to: "mem_04", type: "entity", derived: true },
    { id: "e3", from: "mem_03", to: "mem_04", type: "entity", derived: true },
    { id: "e4", from: "mem_02", to: "mem_06", type: "subject", derived: true },
    { id: "e5", from: "mem_02", to: "mem_07", type: "entity", derived: true },
    { id: "e6", from: "mem_08", to: "mem_04", type: "entity", derived: true },
    { id: "e7", from: "mem_05", to: "mem_06", type: "entity", derived: true },
  ],
  clusters: [
    { id: "c1", type: "subject", label: "User Preferences", member_ids: ["mem_01", "mem_05"], count: 2 },
    { id: "c2", type: "entity", label: "ContextMemory Architecture", member_ids: ["mem_02", "mem_04", "mem_06", "mem_07"], count: 4 },
  ],
};

const INITIAL_STATE: AppState = {
  isOnboarded: localStorage.getItem("cm_onboarded") === "true",
  connection: { phase: "live", ws: "live", engine: "ok", last_error: "" },
  serverUrl: localStorage.getItem("cm_url") || "http://localhost:8765",
  apiKey: localStorage.getItem("cm_key") || "",
  ollamaEndpoint: localStorage.getItem("cm_ollama_url") || "http://localhost:11434",
  activeModel: localStorage.getItem("cm_model") || "qwen3:4b",
  availableModels: ["qwen3:4b", "llama3.2:3b", "gemma2:9b"],
  activeSpaceId: "brain",
  graphData: MOCK_GRAPH,
  selectedNodeId: null,
  selectedNode: null,
  activeQuery: "",
  activeTrace: null,
  isSearching: false,
  queryHistory: [],
  metrics: {
    health: { status: "ok", service: "contextmemory", version: "2.0.0", mode: "local", uptime_ms: 3600000, time: Date.now() },
    stats: { spaces: 1, memories: 8, active: 8, episodes: 12, generation: 1 },
    latency: { count: 42, p50_ms: 0.8, p95_ms: 2.1, avg_ms: 1.1, history_ms: [1.2, 0.8, 0.9, 1.5, 0.7] },
    jobs: { queued: 0, completed: 18, failed: 0 },
    counts: { active: 8, total: 8, episodes: 12, spaces: 1 },
  },
  settingsOpen: false,
  historyModalOpen: false,
};

export class Store<S> {
  private state: S;
  private listeners = new Set<() => void>();

  constructor(initial: S) {
    this.state = initial;
  }

  getState(): S {
    return this.state;
  }

  setState(partial: Partial<S> | ((prev: S) => Partial<S>)): void {
    const patch = typeof partial === "function" ? partial(this.state) : partial;
    this.state = { ...this.state, ...patch };
    this.emit();
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  private emit(): void {
    for (const l of this.listeners) l();
  }
}

export const appStore = new Store<AppState>(INITIAL_STATE);

export function useStore<S, T>(store: Store<S>, selector: (s: S) => T): T {
  const ref = useRef<T | undefined>(undefined);
  const getSnapshot = useCallback(() => {
    const next = selector(store.getState());
    if (!Object.is(next, ref.current)) ref.current = next;
    return ref.current as T;
  }, [store, selector]);
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
}

export const sel = <S, T>(fn: (s: S) => T): ((s: S) => T) => (s: S) => fn(s);
