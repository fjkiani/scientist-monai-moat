// v0.3.0-alpha DemoSamplesTab.
//
// When the deployment is in DEMO_MODE the raw POST endpoints are 403'd,
// but four pre-computed real inference envelopes are served under
// /v1/demo/samples/{screening,biopsy,nsclc,case_full}. This tab reads
// all four and lays out what each pipeline produced end-to-end, with
// full provenance:
//   - which weights ran (Modal MedSigLIP-448, CBIS-DDSM probe,
//     ClinicalBERT, MONAI LUNA16 RetinaNet, NCCN-lite rules,
//     offline Co-Scientist Elo loop)
//   - the sha256 of the input DICOM (proof it was actually loaded)
//   - the model_state each stage reported (loaded_medsiglip,
//     fused_regex_clinicalbert, loaded_luna16_retinanet, etc.)
//   - warm vs cold latency
//   - a plain-English note explaining what was real vs synthetic
//
// A visitor who wants to run the API on their own data clicks the
// "Contact" placeholder which routes to /health.contact_url.

import { useEffect, useMemo, useState } from "react";
import {
  getDemoSampleIndex,
  getDemoSampleScreening,
  getDemoSampleBiopsy,
  getDemoSampleNsclc,
  getDemoSampleCaseFull,
  type DemoSampleIndexResponse,
  type DemoScreeningSample,
  type DemoBiopsySample,
  type DemoNsclcSample,
  type DemoCaseFullSample,
  type DemoProvenance,
} from "../api";

type LoadState<T> =
  | { kind: "loading" }
  | { kind: "ok"; data: T }
  | { kind: "err"; msg: string };

export function DemoSamplesTab() {
  const [index, setIndex] = useState<LoadState<DemoSampleIndexResponse>>({ kind: "loading" });
  const [screening, setScreening] = useState<LoadState<DemoScreeningSample>>({ kind: "loading" });
  const [biopsy, setBiopsy] = useState<LoadState<DemoBiopsySample>>({ kind: "loading" });
  const [nsclc, setNsclc] = useState<LoadState<DemoNsclcSample>>({ kind: "loading" });
  const [caseFull, setCaseFull] = useState<LoadState<DemoCaseFullSample>>({ kind: "loading" });

  useEffect(() => {
    getDemoSampleIndex().then(
      (d) => setIndex({ kind: "ok", data: d }),
      (e) => setIndex({ kind: "err", msg: String(e) }),
    );
    getDemoSampleScreening().then(
      (d) => setScreening({ kind: "ok", data: d }),
      (e) => setScreening({ kind: "err", msg: String(e) }),
    );
    getDemoSampleBiopsy().then(
      (d) => setBiopsy({ kind: "ok", data: d }),
      (e) => setBiopsy({ kind: "err", msg: String(e) }),
    );
    getDemoSampleNsclc().then(
      (d) => setNsclc({ kind: "ok", data: d }),
      (e) => setNsclc({ kind: "err", msg: String(e) }),
    );
    getDemoSampleCaseFull().then(
      (d) => setCaseFull({ kind: "ok", data: d }),
      (e) => setCaseFull({ kind: "err", msg: String(e) }),
    );
  }, []);

  const contactUrl = useMemo(
    () => (index.kind === "ok" ? index.data.contact_url : "https://crispro.ai/contact"),
    [index],
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      <IntroCard index={index} contactUrl={contactUrl} />
      <ScreeningCard state={screening} contactUrl={contactUrl} />
      <BiopsyCard state={biopsy} contactUrl={contactUrl} />
      <NsclcCard state={nsclc} contactUrl={contactUrl} />
      <CaseFullCard state={caseFull} contactUrl={contactUrl} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Intro card

function IntroCard({
  index,
  contactUrl,
}: {
  index: LoadState<DemoSampleIndexResponse>;
  contactUrl: string;
}) {
  return (
    <div className="card">
      <h2>Demo samples — real inference outputs</h2>
      <p>
        This tab shows four pre-computed responses from Oncology Arbiter
        v0.3.0-alpha. Each was captured by hitting the corresponding real
        endpoint on our workers, with the real model weights loaded, on
        real (or clearly-labeled synthetic) inputs. The envelopes below
        are the exact JSON you would receive if the endpoints were live.
      </p>
      <p>
        This deployment is read-only; live inference is disabled. If you
        want to run the API on your own DICOMs and pathology text, use the
        contact link on any tab or here:
      </p>
      <div style={{ marginTop: "0.75rem" }}>
        <a
          href={contactUrl}
          target="_blank"
          rel="noreferrer"
          className="button-like"
          style={{
            display: "inline-block",
            padding: "0.5rem 0.9rem",
            border: "1px solid var(--accent)",
            borderRadius: 4,
            color: "var(--accent)",
            textDecoration: "none",
            fontWeight: 600,
          }}
          data-testid="demo-samples-tab-contact"
        >
          Run on your own data → Contact
        </a>
      </div>
      {index.kind === "ok" && (
        <details style={{ marginTop: "1rem", fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          <summary>Samples available on this deployment ({index.data.samples.length})</summary>
          <ul style={{ marginTop: "0.5rem" }}>
            {index.data.samples.map((s) => (
              <li key={s.kind}>
                <code style={{ fontFamily: "Menlo, monospace" }}>{s.path}</code> — {s.size_bytes} bytes
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Screening card — MedSigLIP-448 (Modal) + CBIS-DDSM logreg probe

function ScreeningCard({
  state,
  contactUrl,
}: {
  state: LoadState<DemoScreeningSample>;
  contactUrl: string;
}) {
  return (
    <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
      <SampleHeader
        kind="screening"
        title="Screening — MedSigLIP-448 + CBIS-DDSM probe"
        pipeline="/v1/screening/analyze"
        state={state}
        contactUrl={contactUrl}
      />
      {state.kind === "ok" && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
            <MetricBox label="model state" value={state.data.provenance.model_state} />
            <MetricBox label="model name" value={state.data.provenance.model_name} />
            <MetricBox label="overall score" value={fmtScore(state.data.overall_score)} />
            <MetricBox label="findings" value={String(state.data.findings?.length ?? 0)} />
          </div>
          <details style={{ marginTop: "1rem" }}>
            <summary>Findings ({state.data.findings?.length ?? 0})</summary>
            <table style={{ width: "100%", fontSize: "0.85rem", marginTop: "0.5rem" }}>
              <thead><tr>
                <th style={{ textAlign: "left" }}>label / prompt</th>
                <th style={{ textAlign: "right" }}>score</th>
              </tr></thead>
              <tbody>
                {(state.data.findings ?? []).map((f, i) => (
                  <tr key={i}>
                    <td style={{ fontFamily: "Menlo, monospace", fontSize: "0.75rem" }}>{f.label}</td>
                    <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>{fmtScore((f as unknown as { score?: number }).score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
          <WarningsList warnings={state.data.warnings} />
          <ProvenanceBlock prov={state.data.demo_provenance} />
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Biopsy card — ClinicalBERT NER + regex fusion

function BiopsyCard({
  state,
  contactUrl,
}: {
  state: LoadState<DemoBiopsySample>;
  contactUrl: string;
}) {
  return (
    <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
      <SampleHeader
        kind="biopsy"
        title="Biopsy — ClinicalBERT + regex fusion"
        pipeline="/v1/biopsy/analyze"
        state={state}
        contactUrl={contactUrl}
      />
      {state.kind === "ok" && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
            <MetricBox label="model state" value={state.data.provenance.model_state} />
            <MetricBox label="model name" value={state.data.provenance.model_name} />
            {"parser_id" in state.data && <MetricBox label="parser id" value={String((state.data as unknown as { parser_id?: string }).parser_id ?? "—")} />}
            {"fusion_mode" in state.data && <MetricBox label="fusion mode" value={String((state.data as unknown as { fusion_mode?: string }).fusion_mode ?? "—")} />}
          </div>
          <details style={{ marginTop: "1rem" }}>
            <summary>Receptor panel + per-field source</summary>
            <ReceptorPanelTable data={state.data as unknown as Record<string, unknown>} />
          </details>
          <WarningsList warnings={state.data.warnings} />
          <ProvenanceBlock prov={state.data.demo_provenance} />
        </>
      )}
    </div>
  );
}

function ReceptorPanelTable({ data }: { data: Record<string, unknown> }) {
  // The biopsy envelope's receptor fields sit at various keys — try each
  // common path defensively. The demo_provenance block on this sample
  // enumerates per_field_source/confidence for what we care about.
  const panel = (data.receptor_panel ?? {}) as Record<string, unknown>;
  const ext = (data.extended_fields ?? {}) as Record<string, unknown>;
  const perSource = (data.per_field_source ?? {}) as Record<string, unknown>;
  const perConf = (data.per_field_confidence ?? {}) as Record<string, unknown>;
  const fields = Array.from(new Set<string>([
    ...Object.keys(panel),
    ...Object.keys(ext),
    ...Object.keys(perSource),
  ]));
  if (fields.length === 0) return <div style={{ color: "var(--fg-muted)", fontSize: "0.85rem" }}>No receptor panel returned.</div>;
  return (
    <table style={{ width: "100%", fontSize: "0.85rem", marginTop: "0.5rem" }}>
      <thead><tr>
        <th style={{ textAlign: "left" }}>field</th>
        <th style={{ textAlign: "left" }}>value</th>
        <th style={{ textAlign: "left" }}>source</th>
        <th style={{ textAlign: "right" }}>conf</th>
      </tr></thead>
      <tbody>
        {fields.map((k) => {
          const v = panel[k] ?? ext[k];
          const s = perSource[k];
          const c = perConf[k];
          return (
            <tr key={k}>
              <td style={{ fontFamily: "Menlo, monospace" }}>{k}</td>
              <td style={{ fontFamily: "Menlo, monospace" }}>{v == null ? "—" : String(v)}</td>
              <td style={{ fontFamily: "Menlo, monospace" }}>{s == null ? "—" : String(s)}</td>
              <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>{c == null ? "—" : Number(c).toFixed(3)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// NSCLC card — MONAI RetinaNet LUNA16

function NsclcCard({
  state,
  contactUrl,
}: {
  state: LoadState<DemoNsclcSample>;
  contactUrl: string;
}) {
  return (
    <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
      <SampleHeader
        kind="nsclc"
        title="NSCLC — MONAI LUNA16 RetinaNet"
        pipeline="/v1/case/full?cancer=nsclc"
        state={state}
        contactUrl={contactUrl}
      />
      {state.kind === "ok" && (
        <>
          {state.data.nsclc && (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
                <MetricBox label="model state" value={state.data.nsclc.model_state} />
                <MetricBox label="bundle" value={state.data.nsclc.luna16?.bundle_version ?? "—"} />
                <MetricBox label="detections" value={String(state.data.nsclc.luna16?.n_detections ?? 0)} />
                <MetricBox label="top score" value={fmtScore(state.data.nsclc.luna16?.top_score)} />
                <MetricBox label="inference (s)" value={fmtSecs(state.data.nsclc.luna16?.inference_seconds)} />
                <MetricBox label="risk score" value={fmtScore(state.data.nsclc.risk_score)} />
              </div>
              {state.data.nsclc.luna16?.detections && state.data.nsclc.luna16.detections.length > 0 && (
                <details style={{ marginTop: "1rem" }}>
                  <summary>Detections ({state.data.nsclc.luna16.detections.length})</summary>
                  <table style={{ width: "100%", fontSize: "0.8rem", marginTop: "0.5rem" }}>
                    <thead><tr>
                      <th>#</th>
                      <th style={{ textAlign: "right" }}>score</th>
                      <th style={{ textAlign: "right" }}>diameter (mm)</th>
                      <th style={{ textAlign: "right" }}>W×H×D (mm)</th>
                    </tr></thead>
                    <tbody>
                      {state.data.nsclc.luna16.detections.map((d, i) => (
                        <tr key={i}>
                          <td style={{ fontFamily: "Menlo, monospace" }}>{i + 1}</td>
                          <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>{fmtScore(d.score)}</td>
                          <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>{fmtSecs(d.diameter_mm)}</td>
                          <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>
                            {fmtSecs(d.width_mm)} × {fmtSecs(d.height_mm)} × {fmtSecs(d.depth_mm)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              )}
            </>
          )}
          <WarningsList warnings={state.data.warnings} />
          <ProvenanceBlock prov={state.data.demo_provenance} />
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Composite case card — screening + biopsy + therapy + Co-Scientist Elo

function CaseFullCard({
  state,
  contactUrl,
}: {
  state: LoadState<DemoCaseFullSample>;
  contactUrl: string;
}) {
  return (
    <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
      <SampleHeader
        kind="case_full"
        title="Case (end-to-end) — screening → biopsy → therapy → Co-Scientist Elo"
        pipeline="/v1/case/full"
        state={state}
        contactUrl={contactUrl}
      />
      {state.kind === "ok" && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
            <MetricBox label="screening state" value={state.data.screening?.provenance?.model_state ?? "—"} />
            <MetricBox label="biopsy state" value={state.data.biopsy?.provenance?.model_state ?? "—"} />
            <MetricBox label="therapy state" value={state.data.therapy?.provenance?.model_state ?? "—"} />
            <MetricBox label="Elo hypotheses" value={String(state.data.elo_ranked_hypotheses?.length ?? 0)} />
          </div>
          {state.data.therapy?.recommended_options && state.data.therapy.recommended_options.length > 0 && (
            <details style={{ marginTop: "1rem" }} open>
              <summary>Therapy — recommended regimens ({state.data.therapy.recommended_options.length})</summary>
              <ul style={{ marginTop: "0.5rem" }}>
                {state.data.therapy.recommended_options.map((opt, i) => {
                  const o = opt as unknown as {
                    regimen?: string; line_of_therapy?: number; rationale?: string;
                    evidence?: Array<{ url?: string; quoted_text?: string; source?: string }>;
                  };
                  return (
                    <li key={i} style={{ marginBottom: "0.5rem" }}>
                      <strong>{o.regimen ?? "—"}</strong>
                      {o.line_of_therapy != null && <span style={{ color: "var(--fg-muted)" }}> · line {o.line_of_therapy}</span>}
                      <div style={{ fontSize: "0.85rem", marginTop: "0.15rem" }}>{o.rationale}</div>
                      {o.evidence && o.evidence.length > 0 && (
                        <div style={{ fontSize: "0.75rem", color: "var(--fg-muted)", marginTop: "0.15rem" }}>
                          {o.evidence.map((e, j) => (
                            <span key={j} style={{ marginRight: "0.5rem" }}>
                              {e.url ? (
                                <a href={e.url} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
                                  {e.quoted_text ?? e.url}
                                </a>
                              ) : (e.quoted_text ?? "")}
                              {e.source && <span> ({e.source})</span>}
                            </span>
                          ))}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </details>
          )}
          {state.data.elo_ranked_hypotheses && state.data.elo_ranked_hypotheses.length > 0 && (
            <details style={{ marginTop: "0.5rem" }}>
              <summary>Co-Scientist Elo hypotheses ({state.data.elo_ranked_hypotheses.length})</summary>
              <table style={{ width: "100%", fontSize: "0.8rem", marginTop: "0.5rem" }}>
                <thead><tr>
                  <th>#</th>
                  <th style={{ textAlign: "left" }}>stage</th>
                  <th style={{ textAlign: "left" }}>statement</th>
                  <th style={{ textAlign: "right" }}>rating</th>
                  <th style={{ textAlign: "right" }}>W/L/D</th>
                </tr></thead>
                <tbody>
                  {state.data.elo_ranked_hypotheses.map((h, i) => {
                    const hyp = h as unknown as {
                      hyp_id?: string; stage?: string; statement?: string;
                      rating?: number; wins?: number; losses?: number; draws?: number;
                    };
                    return (
                      <tr key={i}>
                        <td style={{ fontFamily: "Menlo, monospace" }}>{i + 1}</td>
                        <td style={{ fontFamily: "Menlo, monospace", fontSize: "0.75rem" }}>{hyp.stage}</td>
                        <td>{hyp.statement}</td>
                        <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>{hyp.rating ?? "—"}</td>
                        <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>
                          {hyp.wins ?? 0}/{hyp.losses ?? 0}/{hyp.draws ?? 0}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </details>
          )}
          <WarningsList warnings={state.data.warnings} />
          <ProvenanceBlock prov={state.data.demo_provenance} />
        </>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Shared subcomponents

function SampleHeader({
  kind,
  title,
  pipeline,
  state,
  contactUrl,
}: {
  kind: string;
  title: string;
  pipeline: string;
  state: LoadState<unknown>;
  contactUrl: string;
}) {
  return (
    <>
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.75rem", flexWrap: "wrap" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <span className="pill" style={{ fontFamily: "Menlo, monospace", fontSize: "0.7rem" }}>
          {pipeline}
        </span>
      </div>
      {state.kind === "loading" && <div style={{ color: "var(--fg-muted)", marginTop: "0.5rem" }}>Loading sample…</div>}
      {state.kind === "err" && (
        <div style={{ marginTop: "0.5rem", padding: "0.5rem", borderLeft: "2px solid #d33", background: "#fee" }}>
          Failed to load sample: <code>{state.msg}</code>
          <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
            The demo samples may not be enabled on this deployment.{" "}
            <a href={contactUrl} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
              Contact
            </a> to run the API on your own data.
          </div>
        </div>
      )}
      <input type="hidden" data-testid={`demo-samples-${kind}-marker`} />
    </>
  );
}

function MetricBox({ label, value }: { label: string; value: string | undefined | null }) {
  return (
    <div style={{ padding: "0.5rem 0.75rem", border: "1px solid var(--border)", borderRadius: 4, background: "var(--bg-soft)" }}>
      <div style={{ fontSize: "0.7rem", color: "var(--fg-muted)", textTransform: "uppercase", letterSpacing: "0.03em" }}>{label}</div>
      <div style={{ fontFamily: "Menlo, monospace", fontWeight: 600, marginTop: "0.15rem", wordBreak: "break-all" }}>{value ?? "—"}</div>
    </div>
  );
}

function WarningsList({ warnings }: { warnings: string[] | undefined | null }) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <details style={{ marginTop: "1rem" }}>
      <summary>Warnings ({warnings.length}) — honesty markers</summary>
      <ul style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--fg-muted)" }}>
        {warnings.map((w, i) => (
          <li key={i} style={{ marginBottom: "0.35rem" }}>{w}</li>
        ))}
      </ul>
    </details>
  );
}

function ProvenanceBlock({ prov }: { prov: DemoProvenance | undefined }) {
  if (!prov) return null;
  return (
    <details style={{ marginTop: "1rem", background: "var(--bg-soft)", padding: "0.5rem 0.75rem", borderRadius: 4 }}>
      <summary>demo_provenance — how this sample was generated</summary>
      <div style={{ fontSize: "0.85rem", marginTop: "0.5rem" }}>
        <div>
          <strong>Worker:</strong> <code style={{ fontFamily: "Menlo, monospace" }}>{prov.worker}</code>
          {" · "}
          <strong>Commit:</strong> <code style={{ fontFamily: "Menlo, monospace" }}>{prov.generated_on_commit?.slice(0, 12)}</code>
          {" · "}
          <strong>At:</strong> <span style={{ fontFamily: "Menlo, monospace" }}>{prov.generated_at}</span>
        </div>
        {prov.latency_seconds != null && (
          <div style={{ marginTop: "0.25rem" }}><strong>Latency:</strong> {fmtSecs(prov.latency_seconds)}s</div>
        )}
        {prov.latency_seconds_warm != null && (
          <div style={{ marginTop: "0.25rem" }}>
            <strong>Latency (warm):</strong> {fmtSecs(prov.latency_seconds_warm)}s
            {prov.latency_seconds_cold_first_run != null && <> · <strong>Cold:</strong> {fmtSecs(prov.latency_seconds_cold_first_run)}s</>}
          </div>
        )}
        {prov.notes && <div style={{ marginTop: "0.5rem" }}>{prov.notes}</div>}
        <details style={{ marginTop: "0.5rem" }}>
          <summary>Weights ({prov.weights?.length ?? 0})</summary>
          <ul style={{ marginTop: "0.5rem" }}>
            {(prov.weights ?? []).map((w, i) => (
              <li key={i} style={{ marginBottom: "0.35rem" }}>
                <code style={{ fontFamily: "Menlo, monospace" }}>{w.role}</code>
                {" · "}
                {w.backend}
                {w.endpoint && <> · <code style={{ fontFamily: "Menlo, monospace" }}>{w.endpoint}</code></>}
                {w.artifact && <> · <code style={{ fontFamily: "Menlo, monospace" }}>{w.artifact}</code></>}
                {w.bundle_version && <> · <code style={{ fontFamily: "Menlo, monospace" }}>{w.bundle_version}</code></>}
                {w.model_repo && <> · <code style={{ fontFamily: "Menlo, monospace" }}>{w.model_repo}</code></>}
                {w.spec && <div style={{ fontSize: "0.75rem", color: "var(--fg-muted)", marginTop: "0.15rem" }}>{w.spec}</div>}
              </li>
            ))}
          </ul>
        </details>
        {prov.input && (
          <details style={{ marginTop: "0.5rem" }}>
            <summary>Input metadata</summary>
            <pre style={{ marginTop: "0.5rem", fontSize: "0.75rem", overflow: "auto" }}>
              {JSON.stringify(prov.input, null, 2)}
            </pre>
          </details>
        )}
        {prov.metrics && (
          <details style={{ marginTop: "0.5rem" }}>
            <summary>Reported metrics</summary>
            <pre style={{ marginTop: "0.5rem", fontSize: "0.75rem", overflow: "auto" }}>
              {JSON.stringify(prov.metrics, null, 2)}
            </pre>
          </details>
        )}
      </div>
    </details>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Formatting helpers

function fmtScore(x: number | undefined | null): string {
  if (x == null) return "—";
  return x.toFixed(4);
}

function fmtSecs(x: number | undefined | null): string {
  if (x == null) return "—";
  return x.toFixed(2);
}
