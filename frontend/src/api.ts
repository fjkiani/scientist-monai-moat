// Typed API client for the oncology-arbiter FastAPI service.
// Every response envelope MUST carry disclaimer + caveat + provenance +
// warnings. The client refuses to strip them.

export type ModelState =
  | "placeholder" | "loaded" | "loading" | "unavailable" | "cached"
  | "gated" | "proxy_siglip" | "loaded_medsiglip"
  | "loaded_biopsy_probe" | "loaded_monai_detector"
  | "proxy_monai_heuristic" | "proxy_rules_lite" | "loaded_txgemma"
  // v0.2.2: honest state names surfaced in /health.
  | "proxy_lung_heuristic"        // NSCLC HU-threshold + connected-components
  | "template"                    // L3 arbiter JSON templates (n_training=0)
  | "proxy_regex_v0"              // pathology-report regex parser (stateless)
  | "proxy_co_scientist"          // Co-Scientist Elo tournament (deterministic)
  // v0.3.0: real trained detectors + real ClinicalBERT parser.
  | "loaded_clinicalbert_parser"  // Bio_ClinicalBERT fine-tuned on synthetic corpus
  | "fused_regex_clinicalbert"    // regex ∧ ClinicalBERT fusion
  | "loaded_luna16_retinanet";    // LUNA16-trained MONAI RetinaNet 3D CT detector

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

export type ParseStateValue =
  | "matched"
  | "ambiguous"
  | "no_match"
  | "user_supplied";

export interface ReceptorPanel {
  er_positive: boolean | null;
  pr_positive: boolean | null;
  her2_status: "negative" | "equivocal" | "positive" | null;
  ki67_percent: number | null;
  // v0.2.1: per-field parser provenance. Absent when the backend didn't run
  // the report parser (e.g., WSI-only path with no report_text).
  parse_state?: {
    er: ParseStateValue;
    pr: ParseStateValue;
    her2: ParseStateValue;
    grade: ParseStateValue;
  } | null;
}

export type ParserFieldSource =
  | "regex"
  | "clinicalbert"
  | "fused"
  | "disagreement"
  | "none";

export interface ExtendedReceptorField {
  value: unknown;
  match_state: "matched" | "ambiguous" | "no_match";
  matched_text: string | null;
  span: [number, number] | null;
  confidence: number;
  source: ParserFieldSource;
}

export interface ReportParseBlock {
  parser_id: string;
  fusion_mode: "regex" | "bert" | "fused";
  per_field_confidence: Record<string, number>;
  per_field_source: Record<string, ParserFieldSource>;
  extended_fields: Record<string, ExtendedReceptorField>;
}

export interface BiopsyResponse extends Envelope {
  subtype_prediction: string | null;
  receptor_panel: ReceptorPanel;
  grade: number | null;
  confidence: number | null;
  arbiter_score: ArbiterScore | null;
  // v0.3.0: describes which parser produced the receptor panel + any
  // extended fields (ki67_pct, tumor_size_mm, T/N/M, margin, LVI).
  // Absent for requests that did not include free-text report_text.
  report_parse?: ReportParseBlock | null;
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
  // v0.2.1: rules-lite provenance surfaced in TherapyTab honesty banner.
  // Optional because non-rules paths (TxGemma, placeholder) may omit them.
  branch_id?: string | null;
  rules_sha256?: string | null;
  rules_model_id?: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  disclaimer: string;
  caveat: string;
  endpoints: string[];
  models_loaded: Record<string, ModelState>;
  // v0.3.0-alpha: demo-mode deployment gate. When true, all POSTs return
  // HTTP 403 with contact_url; pre-computed samples served under
  // /v1/demo/samples/{kind}.
  demo_mode?: boolean;
  contact_url?: string | null;
  demo_samples?: string[];
}

// Cancer selector — mirrors backend /v1/case/full?cancer=…
export type CancerId = "breast" | "nsclc";

// v0.3.0: LUNA16 RetinaNet output on the NSCLC response.
export interface Luna16Detection {
  center_z_mm: number;
  center_y_mm: number;
  center_x_mm: number;
  width_mm: number;
  height_mm: number;
  depth_mm: number;
  diameter_mm: number;
  score: number;
}

export interface Luna16DetectionBlock {
  bundle_version: string;
  n_detections: number;
  top_score: number;
  detections: Luna16Detection[];
  inference_seconds: number;
  preprocessing_summary: Record<string, unknown>;
}

export interface NsclcCandidate {
  id: string;
  volume_mm3: number;
  centroid_zyx: [number, number, number];
  bbox_zyx: [number, number, number, number, number, number];
}

export interface NsclcTherapyOption {
  regimen: string;
  line_of_therapy: number;
  rationale: string;
  nccn_section: string;
}

export interface NsclcResponse {
  model_state: ModelState;
  model_name: string;
  warnings: string[];
  lung_voxel_fraction: number | null;
  n_candidates_total: number | null;
  n_candidates_kept: number | null;
  max_diameter_mm: number | null;
  candidates: NsclcCandidate[];
  luna16?: Luna16DetectionBlock | null;
  risk_score: number | null;
  risk_bucket: string | null;
  driving_feature: string | null;
  logit: number | null;
  therapy_recommended: NsclcTherapyOption[];
  therapy_not_recommended: NsclcTherapyOption[];
  series_dir: string | null;
  n_slices: number | null;
  read_seconds: number | null;
  heuristic_seconds: number | null;
}

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

// v0.2.2: /v1/demo/case
export interface DemoCaseResponse {
  dicom_bytes_b64: string;
  dicom_source: string;
  dicom_sha256: string;
  dicom_size_bytes: number;
  report_text: string;
  patient_context: {
    age?: number;
    menopausal_status?: "pre" | "post" | "peri" | "unknown";
    stage_ct?: string;
    grade?: number;
    notes?: string;
    [k: string]: unknown;
  };
  warnings: string[];
}

export function getDemoCase() {
  return get<DemoCaseResponse>("/v1/demo/case");
}

// ── v0.3.0-alpha /v1/demo/samples ────────────────────────────────────────
//
// Pre-computed real inference outputs captured on our workers. Each
// envelope carries a `demo_provenance` sub-block with weights used,
// DICOM sha256, latency, and a plain-English note on what was real vs
// synthetic. See src/oncology_arbiter/api/static/demo_samples/*.json.

export type DemoSampleKind = "screening" | "biopsy" | "nsclc" | "case_full";

export interface DemoProvenanceWeight {
  role: string;
  backend: string;
  endpoint?: string;
  model_repo?: string;
  bundle_version?: string;
  bundle_path?: string;
  artifact?: string;
  size_bytes?: number;
  torchscript?: string;
  pytorch?: string;
  bundle_source?: string;
  license?: string;
  spec?: string;
  embedding_dim?: number;
  device?: string;
  fusion_mode?: string;
  [k: string]: unknown;
}

export interface DemoProvenance {
  generated_at: string;
  generated_on_commit: string;
  worker: string;
  input: Record<string, unknown>;
  weights: DemoProvenanceWeight[];
  metrics?: Record<string, unknown>;
  latency_seconds?: number;
  latency_seconds_warm?: number;
  latency_seconds_cold_first_run?: number;
  luna16_inference_seconds?: number;
  n_detections?: number;
  top_detection_score?: number;
  notes: string;
  model_states?: Record<string, string | null>;
  n_elo_hypotheses?: number;
  n_therapy_options_recommended?: number;
  [k: string]: unknown;
}

export interface DemoSampleIndexEntry {
  kind: DemoSampleKind;
  path: string;
  size_bytes: number;
}

export interface DemoSampleIndexResponse {
  samples: DemoSampleIndexEntry[];
  demo_mode: boolean;
  contact_url: string;
  note: string;
}

export function getDemoSampleIndex() {
  return get<DemoSampleIndexResponse>("/v1/demo/samples");
}

// The concrete envelope for each demo sample kind. Because the wire
// shape mirrors the live endpoints, we reuse existing types and append
// the `demo_provenance` sub-block.
export type DemoScreeningSample = ScreeningResponse & { demo_provenance: DemoProvenance };
export type DemoBiopsySample    = BiopsyResponse   & { demo_provenance: DemoProvenance };
export type DemoNsclcSample     = FullCaseResponse & { demo_provenance: DemoProvenance };
export type DemoCaseFullSample  = FullCaseResponse & { demo_provenance: DemoProvenance };

export function getDemoSampleScreening() {
  return get<DemoScreeningSample>("/v1/demo/samples/screening");
}
export function getDemoSampleBiopsy() {
  return get<DemoBiopsySample>("/v1/demo/samples/biopsy");
}
export function getDemoSampleNsclc() {
  return get<DemoNsclcSample>("/v1/demo/samples/nsclc");
}
export function getDemoSampleCaseFull() {
  return get<DemoCaseFullSample>("/v1/demo/samples/case_full");
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
  // v0.3.0: nsclc block populated when ?cancer=nsclc. Includes an optional
  // luna16 detection block (present when RetinaNet actually ran).
  nsclc: NsclcResponse | null;
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
