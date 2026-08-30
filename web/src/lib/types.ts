export type MemoryKind =
  | "world"
  | "preference"
  | "opinion"
  | "experience"
  | "procedure";

export type MemoryStatus =
  | "active"
  | "superseded"
  | "expired"
  | "forgotten"
  | "disputed";

export interface MemorySource {
  episode_id?: number | null;
  begin?: number;
  end?: number;
  ref?: string;
}

export interface Memory {
  id: string;
  space_id: string;
  cell_id: number;
  text: string;
  subject: string;
  predicate: string;
  object: string;
  kind: MemoryKind;
  status: MemoryStatus;
  confidence: number;
  salience: number;
  source: MemorySource;
  valid_from: number;
  valid_until: number;
  observed_at: number;
  root_id: string;
  parent_id: string;
  tags: string[];
  entities: string[];
  created_at: number;
  updated_at: number;
}

export interface GraphNode extends Omit<Memory, "source"> {
  valid_from: number;
  valid_until: number;
  source_ref?: string;
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  type: "updates" | "subject" | "entity";
  derived: boolean;
}

export interface GraphCluster {
  id: string;
  type: "subject" | "entity";
  label: string;
  member_ids: string[];
  count: number;
}

export interface GraphData {
  space_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  clusters: GraphCluster[];
  generation: number;
  node_count: number;
  edge_count: number;
}

export interface Hit {
  rank: number;
  id: string;
  cell_id: number;
  text: string;
  subject: string;
  predicate: string;
  object: string;
  kind: MemoryKind;
  status: MemoryStatus;
  confidence: number;
  salience: number;
  score: number;
  source_ref: string;
  valid_from: number;
  valid_until: number;
  tags: string[];
  index_generation: number;
}

export interface QueryTrace {
  trace_id: string;
  space_id: string;
  query: string;
  created_at: number;
  plan: Record<string, unknown>;
  channels: { channel: string; selected: boolean }[];
  candidates: Hit[];
  evidence: Hit[];
  candidate_count: number;
  returned_count: number;
  tokens: number;
  budget: number;
  sufficient: boolean;
  used_fallback: boolean;
  index_generation: number;
  timings: Record<string, number>;
  answer: string;
  answer_ms: number;
  model_provider: string;
  model_name: string;
}

export interface MetricSnapshot {
  health: {
    status: string;
    service: string;
    version: string;
    mode: string;
    uptime_ms: number;
    time: number;
  };
  stats: {
    spaces: number;
    memories: number;
    active: number;
    episodes: number;
    generation: number;
  };
  latency: {
    count: number;
    p50_ms: number;
    p95_ms: number;
    avg_ms: number;
    history_ms?: number[];
  };
  jobs: { queued: number; completed: number; failed: number };
  counts: { active: number; total: number; episodes: number; spaces: number };
}

export interface ConnectionState {
  phase: "booting" | "unauthenticated" | "connecting" | "live" | "offline";
  ws: "connecting" | "live" | "offline";
  engine: "ok" | "down";
  last_error: string;
}
