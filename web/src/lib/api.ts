export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

const KEY_STORAGE = "cm_key";

export function getApiKey(): string {
  return localStorage.getItem(KEY_STORAGE) || "";
}

export class Api {
  private async request<T>(
    path: string,
    opts: RequestInit = {},
    key: string = getApiKey()
  ): Promise<T> {
    const headers: Record<string, string> = {
      ...(opts.headers as Record<string, string> | undefined),
    };
    if (key) headers["Authorization"] = `Bearer ${key}`;
    if (opts.body && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const res = await fetch(path, { ...opts, headers });
    const text = await res.text();
    let data: unknown;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }
    if (!res.ok) {
      const detail =
        (data as { detail?: string }).detail || `${res.status} ${res.statusText}`;
      throw new ApiError(res.status, detail);
    }
    return data as T;
  }

  get<T>(path: string, key?: string): Promise<T> {
    return this.request<T>(path, { method: "GET" }, key);
  }

  post<T>(path: string, body: unknown, key?: string): Promise<T> {
    return this.request<T>(path, { method: "POST", body: JSON.stringify(body) }, key);
  }

  ask(
    spaceId: string,
    query: string,
    opts: { model?: string } = {}
  ) {
    return this.post<{
      trace_id: string;
      query: string;
      answer: string;
      hits: import("./types").Hit[];
      candidates: import("./types").Hit[];
      trace: Record<string, unknown>;
      tokens: number;
    }>("/v1/ask", { space_id: spaceId, query, ...opts });
  }

  forget(spaceId: string, id: string) {
    return this.post<{ status: string }>(
      `/v1/memories/${encodeURIComponent(id)}/forget`,
      { space_id: spaceId }
    );
  }
}

export const api = new Api();
