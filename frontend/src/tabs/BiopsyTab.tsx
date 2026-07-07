import { useState } from "react";
import {
  analyzeBiopsy, reasonTherapy,
  type BiopsyResponse, type TherapyResponse,
  type ParserFieldSource,
} from "../api";
import { EnvelopeCard } from "../components/Envelope";
import { ReceptorPanelForm, type ConfirmedPanel } from "../components/ReceptorPanelForm";
import { bytesToBase64Chunked } from "../lib/b64";

/** Colour pill for a source badge on the parse-detail card. */
function SourcePill({ source }: { source: ParserFieldSource }) {
  const colour: Record<ParserFieldSource, string> = {
    fused: "#75A025",           // green — regex ∧ BERT agree
    regex: "#0279EE",           // blue — regex-only
    clinicalbert: "#FD9BED",    // pink — BERT-only
    disagreement: "#FF9400",    // orange — regex ≠ BERT
    none: "#94a3b8",            // grey — no signal
  };
  return (
    <span style={{
      display: "inline-block", padding: "0.05rem 0.4rem",
      borderRadius: 3, background: colour[source], color: "#000",
      fontSize: "0.7rem", fontWeight: 600,
    }}>
      {source}
    </span>
  );
}

/** Report-parse detail card (v0.3.0).
 *  - Always renders if `result.report_parse` is present.
 *  - Colour-codes each core field by source (fused/regex/BERT/disagreement).
 *  - Lists extended fields (ki67_pct, tumor_size_mm, T/N/M, margin, LVI)
 *    below — these ONLY surface when a BERT or fused parser ran. */
function ReportParseCard({ result }: { result: BiopsyResponse }) {
  const rp = result.report_parse;
  if (!rp) return null;
  const core = ["er", "pr", "her2", "grade"] as const;
  const ext = Object.entries(rp.extended_fields);
  const isFused = rp.fusion_mode === "fused";
  const isBert = rp.fusion_mode === "bert";
  return (
    <div className="card">
      <h2>Report parse detail</h2>
      <div style={{ fontSize: "0.85rem", color: "var(--fg-muted)",
                    marginBottom: "0.5rem" }}>
        parser: <code>{rp.parser_id}</code>
        {" · mode: "}<code>{rp.fusion_mode}</code>
        {isFused && (
          <span style={{ marginLeft: "0.5rem",
                         background: "#f0fdf4", padding: "0.05rem 0.4rem",
                         border: "1px solid #86efac", borderRadius: 3 }}>
            regex ∧ ClinicalBERT (both must agree to auto-fill)
          </span>
        )}
        {isBert && (
          <span style={{ marginLeft: "0.5rem",
                         background: "#fef3c7", padding: "0.05rem 0.4rem",
                         border: "1px solid #f59e0b", borderRadius: 3 }}>
            ClinicalBERT-only (uncalibrated, synthetic-trained)
          </span>
        )}
      </div>
      <table style={{ width: "100%", fontSize: "0.85rem",
                      borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "0.25rem" }}>Field</th>
            <th style={{ textAlign: "left", padding: "0.25rem" }}>Source</th>
            <th style={{ textAlign: "left", padding: "0.25rem" }}>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {core.map((k) => {
            const src = rp.per_field_source[k] ?? "none";
            const conf = rp.per_field_confidence[k];
            return (
              <tr key={k} style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "0.25rem" }}><code>{k}</code></td>
                <td style={{ padding: "0.25rem" }}>
                  <SourcePill source={src} />
                </td>
                <td style={{ padding: "0.25rem" }}>
                  <code>{conf != null ? conf.toFixed(2) : "—"}</code>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {ext.length > 0 && (
        <>
          <h3 style={{ marginTop: "1rem", fontSize: "0.95rem" }}>
            Extended fields
          </h3>
          <p style={{ fontSize: "0.8rem", color: "var(--fg-muted)",
                       marginBottom: "0.5rem" }}>
            These are BERT-only fields the regex parser cannot produce. Values
            are informational — the therapy branch does not currently depend
            on any of them.
          </p>
          <table style={{ width: "100%", fontSize: "0.85rem",
                          borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "0.25rem" }}>Field</th>
                <th style={{ textAlign: "left", padding: "0.25rem" }}>Value</th>
                <th style={{ textAlign: "left", padding: "0.25rem" }}>State</th>
                <th style={{ textAlign: "left", padding: "0.25rem" }}>Source</th>
                <th style={{ textAlign: "left", padding: "0.25rem" }}>Conf.</th>
              </tr>
            </thead>
            <tbody>
              {ext.map(([name, f]) => (
                <tr key={name} style={{ borderTop: "1px solid var(--border)" }}>
                  <td style={{ padding: "0.25rem" }}><code>{name}</code></td>
                  <td style={{ padding: "0.25rem" }}>
                    <code>{f.value == null ? "—" : String(f.value)}</code>
                  </td>
                  <td style={{ padding: "0.25rem" }}>
                    <code>{f.match_state}</code>
                  </td>
                  <td style={{ padding: "0.25rem" }}>
                    <SourcePill source={f.source} />
                  </td>
                  <td style={{ padding: "0.25rem" }}>
                    <code>{f.confidence.toFixed(2)}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

// Canned luminal-A demo report shipped for the tumor-board dry-run.
// Expected parser output: ER matched (True), PR matched (True),
// HER2 matched (negative), grade matched (2). Therapy branch:
// hr_positive_her2_negative → "Aromatase inhibitor (letrozole/anastrozole)".
const LUMINAL_A_EXAMPLE = `Age: 58, postmenopausal
Stage: T1N0M0
Pathology:
  Invasive ductal carcinoma of the right breast, 1.4 cm.
  Estrogen Receptor: Positive (95%).
  Progesterone Receptor: Positive (80%).
  HER2/neu: Negative (IHC 1+).
  Nottingham Grade: 2.
  Ki-67 index: 12%.`;

export function BiopsyTab() {
  const [file, setFile] = useState<File | null>(null);
  const [reportText, setReportText] = useState("");
  const [busy, setBusy] = useState(false);
  const [therapyBusy, setTherapyBusy] = useState(false);
  const [result, setResult] = useState<BiopsyResponse | null>(null);
  const [therapy, setTherapy] = useState<TherapyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setBusy(true); setErr(null); setResult(null); setTherapy(null);
    try {
      const payload: Record<string, unknown> = {};
      if (file) {
        const bytes = new Uint8Array(await file.arrayBuffer());
        payload.wsi_bytes_b64 = bytesToBase64Chunked(bytes);
      }
      if (reportText.trim()) payload.report_text = reportText.trim();
      const r = await analyzeBiopsy(payload);
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  async function runTherapy(confirmed: ConfirmedPanel) {
    if (!result) return;
    setTherapyBusy(true); setErr(null); setTherapy(null);
    try {
      const t = await reasonTherapy({
        biopsy_output: result,
        receptors_override: {
          er_positive: confirmed.er_positive,
          pr_positive: confirmed.pr_positive,
          her2_status: confirmed.her2_status,
        },
        patient_context: {},
      });
      setTherapy(t);
    } catch (e) { setErr(String(e)); } finally { setTherapyBusy(false); }
  }

  return (
    <>
      <div className="card">
        <h2>Biopsy · WSI image or pathology report</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Upload a whole-slide image (WSI patch, gross photo, or JPEG/PNG) and/or
          paste a pathology report. The L4b MedSigLIP-448 probe returns a subtype
          prediction with a calibrated probability. WSI weights are
          {" "}<strong>synthetic</strong>. The report is parsed for ER/PR/HER2/
          grade — v0.2.1 used a <code>proxy_regex_v0</code>; v0.3.0 optionally
          fuses regex output with a fine-tuned <code>Bio_ClinicalBERT</code>
          token classifier (extended fields: Ki-67, tumor size, T/N/M, margin,
          LVI). <strong>You must confirm every core field</strong> before the
          therapy branch runs, regardless of parser mode.
        </p>
        <label>WSI (image bytes — no OpenSlide parser)</label>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <label style={{ marginTop: "0.75rem" }}>Pathology report (optional free text)</label>
        <textarea value={reportText} onChange={(e) => setReportText(e.target.value)}
                  placeholder="Free-text pathology report…" data-testid="biopsy-report-textarea"/>
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem", flexWrap: "wrap" }}>
          <button className="primary" onClick={submit} disabled={busy || (!file && !reportText.trim())}>
            {busy ? "Analyzing…" : "Analyze biopsy"}
          </button>
          <button
            type="button"
            onClick={() => setReportText(LUMINAL_A_EXAMPLE)}
            style={{ background: "var(--panel)", border: "1px solid var(--border)",
                     padding: "0.5rem 0.75rem", cursor: "pointer" }}
            data-testid="load-luminal-a-example"
          >
            Load ER+ luminal-A example
          </button>
        </div>
        {err && <div className="warning" style={{ marginTop: "0.75rem" }}>{err}</div>}
      </div>

      {result && (
        <>
          <div className="card">
            <h2>Subtype prediction</h2>
            <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
              <div>
                <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Predicted subtype</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>
                  {result.subtype_prediction ?? "—"}
                </div>
              </div>
              <div>
                <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Confidence</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>
                  {result.confidence !== null ? `${(result.confidence * 100).toFixed(1)}%` : "n/a"}
                </div>
              </div>
              <div>
                <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Parser grade</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>
                  {result.grade ?? "—"}
                </div>
              </div>
            </div>
          </div>

          <ReportParseCard result={result} />

          <ReceptorPanelForm
            panel={result.receptor_panel}
            parsedGrade={result.grade}
            onConfirm={runTherapy}
            busy={therapyBusy}
          />

          {therapy && (
            <div className="card">
              <h2>Therapy · rules-lite recommendation</h2>
              <div style={{ background: "#fef3c7", border: "1px solid #f59e0b",
                            padding: "0.75rem", marginBottom: "0.75rem",
                            fontSize: "0.85rem", borderRadius: 6 }}>
                <strong>Research-use disclaimer.</strong> This is a deterministic
                proxy of a small NCCN-lite ruleset (<code>nccn-lite-v0</code>),
                not TxGemma. Treat every recommendation as a discussion prompt,
                not a clinical directive. See the caveat and evidence records
                below for the exact ruleset SHA-256.
              </div>
              {therapy.recommended_options.length === 0 && (
                <p style={{ color: "var(--fg-muted)" }}>No options recommended.</p>
              )}
              {therapy.recommended_options.map((opt, i) => (
                <div key={i} style={{ padding: "0.5rem 0", borderBottom: "1px solid var(--border)" }}>
                  <div style={{ fontWeight: 600 }}>{opt.regimen}</div>
                  <div style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
                    {opt.rationale}
                  </div>
                </div>
              ))}
              <EnvelopeCard env={therapy} arbiter={therapy.arbiter_score} />
            </div>
          )}

          <EnvelopeCard env={result} arbiter={result.arbiter_score} />
        </>
      )}
    </>
  );
}
