import { useState } from "react";
import { analyzeBiopsy, reasonTherapy, type BiopsyResponse, type TherapyResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";
import { ReceptorPanelForm, type ConfirmedPanel } from "../components/ReceptorPanelForm";

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
        payload.wsi_bytes_b64 = btoa(String.fromCharCode(...bytes));
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
          prediction with a calibrated probability. Weights are
          {" "}<strong>synthetic</strong>. The report is parsed for ER/PR/HER2/grade
          by a <code>proxy_regex_v0</code> and shown below — <strong>you must
          confirm every field</strong> before the therapy branch runs.
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
