import type {
  CalculationEnvelope,
  ChatResponse,
  PhaseDiagramResponse,
  TaskManifest,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload = await response.json();
  if (!response.ok) {
    const message = payload?.error?.message ?? payload?.detail ?? "请求失败";
    throw new Error(message);
  }
  return payload as T;
}

export function sendChat(message: string, conversationId?: string): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
}

export function rerunTask(task: TaskManifest): Promise<CalculationEnvelope> {
  const paths: Record<string, string> = {
    isobaric_vle: "isobaric-vle",
    isothermal_vle: "isothermal-vle",
    bubble_point: "bubble-point",
    dew_point: "dew-point",
    tp_flash: "tp-flash",
    azeotrope: "azeotrope",
  };
  const endpoint = paths[task.calculation_type];
  if (!endpoint) throw new Error(`不支持的计算类型：${task.calculation_type}`);
  return request<CalculationEnvelope>(`/api/calculations/${endpoint}`, {
    method: "POST",
    body: JSON.stringify(task),
  });
}

export function phaseDiagram(task: TaskManifest): Promise<PhaseDiagramResponse> {
  return request<PhaseDiagramResponse>("/api/calculations/phase-diagram", {
    method: "POST",
    body: JSON.stringify(task),
  });
}

export function exportUrl(runId: string, format: "json" | "csv" | "dwsim"): string {
  return `${API_URL}/api/runs/${runId}/export?format=${format}`;
}

/**
 * Download a generated .dwxmz file from a relative URI by triggering the
 * browser's native save flow (blob + hidden anchor).  The raw download endpoint
 * URL is consumed only inside this helper, so it is never exposed to the user.
 */
export async function downloadDwsimFile(relativeUri: string, fallbackName: string): Promise<void> {
  const url = relativeUri.startsWith("http") ? relativeUri : `${API_URL}${relativeUri}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`下载失败：HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  // Derive a friendly filename from the URI (e.g. extractive-...dwxmz).
  const segment = relativeUri.split("/").pop() ?? fallbackName;
  link.download = segment;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(objectUrl);
}
