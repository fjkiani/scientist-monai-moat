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

// ── HTTP helpers ─────────────────────────────────────────────────────────
const API_BASE = "";

async function post<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);
  return (await resp.json()) as T;
}

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`);
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
  return get<HealthResponse>("/health");
}

// ── /v1/case/full ────────────────────────────────────────────────────────

export interface FullCaseResponse extends Envelope {
  screening: ScreeningResponse | null;
  biopsy: BiopsyResponse | null;
  therapy: TherapyResponse | null;
  elo_ranked_hypotheses: Array<Record<string, unknown>>;
}

export function runCaseFull(payload: unknown) {
  return post<FullCaseResponse>("/v1/case/full", payload);
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
