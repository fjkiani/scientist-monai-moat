// Typed API client for the oncology-arbiter FastAPI service.
// Every response envelope MUST carry disclaimer + caveat + provenance +
// warnings. The client refuses to strip them.

export type ModelState =
  | "placeholder" | "loaded" | "loading" | "unavailable" | "cached"
  | "gated" | "proxy_siglip" | "loaded_medsiglip"
  | "loaded_biopsy_probe" | "loaded_monai_detector"
  | "proxy_monai_heuristic" | "proxy_rules_lite" | "loaded_txgemma";

export interface Provenance {
  model_state: ModelState;
  model_name?: string | null;
  model_version?: string | null;
  request_id: string;
}

export interface HonestyGateReport {
  seen_urls_count: number;
  evidence_kept: number;
  evidence_dropped: number;
}

export interface EvidenceRecord {
  url: string;
  quoted_text: string;
  source: string;
}

export interface Envelope {
  disclaimer: string;
  caveat: string;
  provenance: Provenance;
  honesty_gate: HonestyGateReport;
  evidence: EvidenceRecord[];
  warnings: string[];
}

export interface ScreeningFinding {
  label: string;
  score: number;
  location_bbox_normalized: [number, number, number, number] | null;
}

export interface ArbiterScore {
  model_name: string;
  p_positive: number;
  logit: number;
  risk_bucket: "LOW" | "MID" | "HIGH";
  recommendation: string;
  term_contributions: Record<string, number>;
  driving_feature: string;
  driving_feature_contribution: number;
  positive_class: string;
  n_training: number;
  model_state: "template" | "frozen";
  caveat: string;
}

export interface ScreeningResponse extends Envelope {
  laterality: "L" | "R" | "U";
  view: "CC" | "MLO" | "U";
  orientation_flipped: boolean;
  breast_mask_coverage: number;
  findings: ScreeningFinding[];
  overall_score: number | null;
  arbiter_score: ArbiterScore | null;
}

export interface BiopsyResponse extends Envelope {
  subtype_prediction: string | null;
  receptor_panel: {
    er_positive: boolean | null;
    pr_positive: boolean | null;
    her2_status: "negative" | "equivocal" | "positive" | null;
    ki67_percent: number | null;
  };
  grade: number | null;
  confidence: number | null;
  arbiter_score: ArbiterScore | null;
}

export interface TherapyOption {
  regimen: string;
  line_of_therapy: number;
  rationale: string;
  evidence: EvidenceRecord[];
  contraindications: string[];
}

export interface TherapyResponse extends Envelope {
  recommended_options: TherapyOption[];
  not_recommended: TherapyOption[];
  arbiter_score: ArbiterScore | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  disclaimer: string;
  caveat: string;
  endpoints: string[];
  models_loaded: Record<string, ModelState>;
}

// Cancer selector — mirrors backend /v1/case/full?cancer=…
export type CancerId = "breast" | "nsclc";

// ── HTTP helpers ─────────────────────────────────────────────────────────
const API_BASE = "";

/** Auth401Error is thrown on 401 so the UI can open the API-key drawer. */
export class Auth401Error extends Error {
  status = 401 as const;
  constructor(public detail: string) { super(`401: ${detail}`); }
}

/** Client-side hooks the SPA installs at boot: injects X-API-Key from
 *  localStorage, captures X-Request-Id from responses, and calls a
 *  `on401` callback so the drawer can pop.
 *  Kept as module-level bindings (not a context) so `post/get` can stay
 *  simple; App.tsx wires it via `installApiHooks()` on mount. */
let _apiKeyProvider: () => string = () => "";
let _requestIdSink: (rid: string) => void = () => { /* no-op */ };
let _on401: (detail: string) => void = () => { /* no-op */ };

export function installApiHooks(opts: {
  apiKey: () => string;
  onRequestId: (rid: string) => void;
  on401: (detail: string) => void;
}): void {
  _apiKeyProvider = opts.apiKey;
  _requestIdSink = opts.onRequestId;
  _on401 = opts.on401;
}

function _headers(extra: HeadersInit = {}): HeadersInit {
  const h: Record<string, string> = { ...(extra as Record<string, string>) };
  const k = _apiKeyProvider();
  if (k) h["X-API-Key"] = k;
  return h;
}

function _captureRequestId(resp: Response): void {
  const rid = resp.headers.get("X-Request-Id");
  if (rid) _requestIdSink(rid);
}

async function _handleUnauthorized(resp: Response): Promise<never> {
  let detail = resp.statusText;
  try { const j = await resp.json(); if (j?.detail) detail = String(j.detail); } catch { /* ignore */ }
  _on401(detail);
  throw new Auth401Error(detail);
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: _headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  _captureRequestId(resp);
  if (resp.status === 401) return _handleUnauthorized(resp);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);
  return (await resp.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, { headers: _headers() });
  _captureRequestId(resp);
  if (resp.status === 401) return _handleUnauthorized(resp);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);
  return (await resp.json()) as T;
}

export function analyzeScreening(payload: unknown) {
  return post<ScreeningResponse>("/v1/screening/analyze", payload);
}
export function analyzeBiopsy(payload: unknown) {
  return post<BiopsyResponse>("/v1/biopsy/analyze", payload);
}
export function reasonTherapy(payload: unknown) {
  return post<TherapyResponse>("/v1/therapy/reason", payload);
}
export function getHealth() {
  return get<HealthResponseWithCancers>("/health");
}

export interface CancerCapability {
  state: string;                 // ModelState value, but comes over as string
  case_full: boolean;
  endpoints: string[];
  notes?: string;
}

export interface HealthResponseWithCancers extends HealthResponse {
  cancers?: Record<string, CancerCapability>;
}

// ── /v1/case/full ────────────────────────────────────────────────────────

export interface FullCaseResponse extends Envelope {
  screening: ScreeningResponse | null;
  biopsy: BiopsyResponse | null;
  therapy: TherapyResponse | null;
  elo_ranked_hypotheses: Array<Record<string, unknown>>;
}

/** `cancer` is a REQUIRED first argument (never defaulted silently); the
 *  wire always carries `?cancer=…` so the audit log ties every response
 *  to the cancer the operator meant to run. */
export function runCaseFull(cancer: CancerId, payload: unknown) {
  return post<FullCaseResponse>(`/v1/case/full?cancer=${encodeURIComponent(cancer)}`, payload);
}

// ── /v1/model-cards ──────────────────────────────────────────────────────

export interface ModelCardSummary {
  slug: string;
  title: string;
  n_bytes: number;
  honesty_markers: Record<string, boolean>;
}
export interface ModelCardsIndex {
  cards: ModelCardSummary[];
}
export function listModelCards() {
  return get<ModelCardsIndex>("/v1/model-cards");
}
