/** CUNCA-Hybrid API client — calls the FastAPI backend. */

export const API_URL = (import.meta.env.PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');
export const API_KEY = import.meta.env.PUBLIC_API_KEY ?? '';

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json', ...extra };
  if (API_KEY) h['Authorization'] = `Bearer ${API_KEY}`;
  return h;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GenerateRequest  { input: string; max_tokens?: number; temperature?: number; }
export interface GenerateResponse { output: string; model: string; tokens_used: number; }
export interface HealthResponse   { status: string; service?: string; on_premises?: boolean; }

export interface AdminSummariseRequest    { text: string; lang?: string; max_sentences?: number; }
export interface AdminSummariseResponse   { summary: string; lang: string; char_count_in: number; on_premises: boolean; }
export interface AdminRegulationRequest   { question: string; regulation_context?: string; lang?: string; }
export interface AdminRegulationResponse  { answer: string; confidence: string; disclaimer: string; lang: string; }

export interface IndustryAnomalyRequest   { sensor_id: string; metric: string; value: number; threshold: number; lang?: string; }
export interface IndustryAnomalyResponse  { description: string; severity: string; recommended_action: string; lang: string; }
export interface IndustryWorkOrderRequest { equipment_id: string; fault_description: string; technician?: string; lang?: string; }
export interface IndustryWorkOrderResponse{ order_id: string; equipment_id: string; description: string; priority: string; lang: string; }

export interface HealthSummariseRequest   { note: string; patient_id?: string; lang?: string; }
export interface HealthSummariseResponse  { summary: string; disclaimer: string; patient_id: string; lang: string; on_premises: boolean; }
export interface ICD10Request             { diagnosis_text: string; lang?: string; }
export interface ICD10Response            { suggestions: { code: string; description: string; keyword_match: string | null }[]; disclaimer: string; lang: string; }

// ---------------------------------------------------------------------------
// Core generate
// ---------------------------------------------------------------------------

export async function generate(req: GenerateRequest): Promise<GenerateResponse> {
  const res = await fetch(`${API_URL}/generate`, {
    method: 'POST', headers: headers(), body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

/** Stream tokens from /generate/stream. Calls onToken for each fragment, returns full text. */
export async function generateStream(
  req: GenerateRequest,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(`${API_URL}/generate/stream`, {
    method: 'POST', headers: headers(), body: JSON.stringify(req), signal,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let full = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split('\n\n');
    buf = parts.pop() ?? '';
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') return full;
      try {
        const obj = JSON.parse(data);
        if (obj.token) { onToken(obj.token); full += obj.token; }
      } catch { /* ignore malformed */ }
    }
  }
  return full;
}

export async function health(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`, { headers: headers() });
  return res.json();
}

export async function metrics(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_URL}/metrics`, { headers: headers() });
  return res.json();
}

// ---------------------------------------------------------------------------
// Demo endpoints
// ---------------------------------------------------------------------------

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: 'POST', headers: headers(), body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const admin = {
  summarise: (req: AdminSummariseRequest)  => post<AdminSummariseResponse>('/demo/admin/summarise', req),
  regulation: (req: AdminRegulationRequest) => post<AdminRegulationResponse>('/demo/admin/regulation', req),
};

export const industry = {
  anomaly:   (req: IndustryAnomalyRequest)   => post<IndustryAnomalyResponse>('/demo/industry/anomaly', req),
  workOrder: (req: IndustryWorkOrderRequest) => post<IndustryWorkOrderResponse>('/demo/industry/workorder', req),
};

export const healthDemo = {
  summarise: (req: HealthSummariseRequest) => post<HealthSummariseResponse>('/demo/health/summarise', req),
  icd10:     (req: ICD10Request)           => post<ICD10Response>('/demo/health/icd10', req),
};
