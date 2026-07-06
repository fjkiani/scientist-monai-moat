import { useState } from "react";
import {
  reasonTherapy,
  runCaseFull,
  type FullCaseResponse,
  type TherapyResponse,
} from "../api";
import { EnvelopeCard } from "../components/Envelope";
import { BboxOverlay } from "../components/BboxOverlay";
import {
  ReceptorPanelForm,
  type ConfirmedPanel,
} from "../components/ReceptorPanelForm";
import { getCancer } from "../settings";

// Canned luminal-A demo report — same as BiopsyTab, kept literal (not
// imported) so each tab's demo affordance is self-documenting.
const LUMINAL_A_EXAMPLE = `Age: 58, postmenopausal
Stage: T1N0M0
Pathology:
  Invasive ductal carcinoma of the right breast, 1.4 cm.
  Estrogen Receptor: Positive (95%).
  Progesterone Receptor: Positive (80%).
  HER2/neu: Negative (IHC 1+).
  Nottingham Grade: 2.
  Ki-67 index: 12%.`;

/**
 * Two-stage tumor-board flow:
 *
 *   Stage 1 · Run full case
 *     Sends the DICOM + WSI + report + patient context to /v1/case/full.
 *     The backend runs screening → biopsy (regex parser) → therapy (using
 *     the parser output) → co-scientist. We show every stage AS IS.
 *
 *   Stage 2 · Confirm receptors → re-run therapy
 *     The pathologist reviews the parser's ER/PR/HER2/grade in the form.
 *     Any correction flips the pill to user_supplied. On Confirm, we
 *     re-issue only /v1/therapy/reason with receptors_override — the
 *     screening + biopsy + co-scientist blocks stay as-is, only the
 *     therapy card is swapped in-place.
 *
 * This keeps the demo honest (parser output is visible AND user-confirmed
 * separately) without doubling the round-trip cost.
 */
export function CaseViewTab() {
  const [dicomFile, setDicomFile] = useState<File | null>(null);
  const [wsiFile, setWsiFile] = useState<File | null>(null);
  const [reportText, setReportText] = useState("");
  const [age, setAge] = useState<number>(58);
  const [menopausal, setMenopausal] = useState<"pre" | "post" | "peri" | "unknown">("post");
  const [stage, setStage] = useState<string>("T1N0M0");

  const [busy, setBusy] = useState(false);
  const [therapyBusy, setTherapyBusy] = useState(false);
  const [result, setResult] = useState<FullCaseResponse | null>(null);
  const [confirmedTherapy, setConfirmedTherapy] = useState<TherapyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  async function submit() {
    setBusy(true); setErr(null); setResult(null); setConfirmedTherapy(null);
    try {
      const body: Record<string, unknown> = {
        therapy_context: {
          age,
          menopausal_status: menopausal,
          prior_therapies: [],
          comorbidities: [],
          genomic_markers: { stage },
        },
      };
      if (dicomFile) {
        const bytes = new Uint8Array(await dicomFile.arrayBuffer());
        body.screening_input = {
          dicom_bytes_b64: btoa(String.fromCharCode(...bytes)),
        };
        setImagePreview(URL.createObjectURL(dicomFile));
      }
      if (wsiFile || reportText.trim()) {
        const bi: Record<string, unknown> = {};
        if (wsiFile) {
          const bytes = new Uint8Array(await wsiFile.arrayBuffer());
          bi.wsi_bytes_b64 = btoa(String.fromCharCode(...bytes));
        }
        if (reportText.trim()) bi.report_text = reportText.trim();
        body.biopsy_input = bi;
      }
      // Case view is breast-specific today (all sub-inputs are mammo + WSI +
      // free-text report). The cancer selector still governs which query
      // param goes on the wire — an operator who switched to nsclc gets
      // routed to <NsclcTab/> by App.tsx before they hit this button.
      const r = await runCaseFull(getCancer(), body);
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  async function reRunTherapyWithOverride(confirmed: ConfirmedPanel) {
    if (!result || !result.biopsy) return;
    setTherapyBusy(true); setErr(null); setConfirmedTherapy(null);
    try {
      const t = await reasonTherapy({
        biopsy_output: result.biopsy,
        receptors_override: {
          er_positive: confirmed.er_positive,
          pr_positive: confirmed.pr_positive,
          her2_status: confirmed.her2_status,
        },
        patient_context: {
          age,
          menopausal_status: menopausal,
          prior_therapies: [],
          comorbidities: [],
          genomic_markers: { stage },
        },
      });
      setConfirmedTherapy(t);
    } catch (e) { setErr(String(e)); } finally { setTherapyBusy(false); }
  }

  // The active therapy result — if user has confirmed, that wins; otherwise
  // fall back to the initial parser-driven therapy block from case_full.
  const activeTherapy: TherapyResponse | null =
    confirmedTherapy ?? (result?.therapy ?? null);

  return (
    <>
      <div className="card">
        <h2>Case view · end-to-end (screening → biopsy → therapy)</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Chains all three stage endpoints server-side and returns the L5
          Elo-ranked hypotheses block. Any stage can be omitted — the server
          will return that section as <code>null</code>. The pathology-report
          receptors are parsed by <code>proxy_regex_v0</code> and displayed
          below; <strong>you must confirm every field</strong> to re-run the
          therapy branch with the values you approve.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
          <div>
            <label>DICOM (screening)</label>
            <input type="file" accept=".dcm,image/*"
                   onChange={(e) => setDicomFile(e.target.files?.[0] || null)} />
          </div>
          <div>
            <label>WSI (biopsy)</label>
            <input type="file" accept="image/*"
                   onChange={(e) => setWsiFile(e.target.files?.[0] || null)} />
          </div>
          <div style={{ gridColumn: "1 / 3" }}>
            <label>Pathology report</label>
            <textarea value={reportText} onChange={(e) => setReportText(e.target.value)}
                      placeholder="Free-text pathology report…"
                      data-testid="caseview-report-textarea" />
          </div>
          <div>
            <label>Age</label>
            <input type="number" min={18} max={120} value={age}
                   onChange={(e) => setAge(Number(e.target.value))} />
          </div>
          <div>
            <label>Menopausal status</label>
            <select value={menopausal} onChange={(e) => setMenopausal(e.target.value as any)}>
              <option>pre</option><option>post</option><option>peri</option><option>unknown</option>
            </select>
          </div>
          <div>
            <label>Stage (T/N/M)</label>
            <input value={stage} onChange={(e) => setStage(e.target.value)} placeholder="T1N0M0" />
          </div>
        </div>

        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem", flexWrap: "wrap" }}>
          <button className="primary" onClick={submit} disabled={busy}>
            {busy ? "Running case…" : "Run full case"}
          </button>
          <button
            type="button"
            onClick={() => setReportText(LUMINAL_A_EXAMPLE)}
            style={{ background: "var(--panel)", border: "1px solid var(--border)",
                     padding: "0.5rem 0.75rem", cursor: "pointer" }}
            data-testid="load-luminal-a-example-caseview"
          >
            Load ER+ luminal-A example
          </button>
        </div>
        {err && <div className="warning" style={{ marginTop: "0.75rem" }}>{err}</div>}
      </div>

      {result && (
        <>
          {result.screening && (
            <div className="card">
              <h2>Screening summary</h2>
              <div>
                <span className="pill">{result.screening.laterality}</span>
                <span className="pill">{result.screening.view}</span>
                <span className={`pill ${result.screening.provenance.model_state}`}>
                  {result.screening.provenance.model_state}
                </span>
                <span className="pill">
                  overall {result.screening.overall_score !== null ? result.screening.overall_score.toExponential(3) : "n/a"}
                </span>
              </div>
              <div style={{ marginTop: "0.5rem" }}>
                <BboxOverlay imageDataUrl={imagePreview} findings={result.screening.findings} width={384} height={384} />
              </div>
              <details style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}>
                <summary>{result.screening.findings.length} findings</summary>
                <pre>{JSON.stringify(result.screening.findings, null, 2)}</pre>
              </details>
            </div>
          )}

          {result.biopsy && (
            <>
              <div className="card">
                <h2>Biopsy summary (parser output)</h2>
                <div>
                  <span className={`pill ${result.biopsy.provenance.model_state}`}>
                    {result.biopsy.provenance.model_state}
                  </span>
                  <span className="pill">subtype: {result.biopsy.subtype_prediction ?? "—"}</span>
                  <span className="pill">grade: {result.biopsy.grade ?? "—"}</span>
                  <span className="pill">
                    confidence: {result.biopsy.confidence !== null ? (result.biopsy.confidence * 100).toFixed(1) + "%" : "—"}
                  </span>
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--fg-muted)", marginTop: "0.5rem" }}>
                  The receptor panel below is what the regex parser saw. Confirm
                  or correct it, then click the Confirm button to re-run the
                  therapy branch with the values you approve.
                </p>
              </div>

              <ReceptorPanelForm
                panel={result.biopsy.receptor_panel}
                parsedGrade={result.biopsy.grade}
                onConfirm={reRunTherapyWithOverride}
                busy={therapyBusy}
              />
            </>
          )}

          {activeTherapy && (
            <div className="card">
              <h2>
                Therapy · {confirmedTherapy ? "user-confirmed" : "parser-driven"}{" "}
                recommendation
              </h2>
              <div style={{ background: "#fef3c7", border: "1px solid #f59e0b",
                            padding: "0.75rem", marginBottom: "0.75rem",
                            fontSize: "0.85rem", borderRadius: 6 }}
                   data-testid="therapy-caveat-banner">
                <strong>Research-use disclaimer.</strong> {confirmedTherapy
                  ? "This card reflects the values YOU confirmed. "
                  : "This card is driven by the raw parser output — click Confirm above to override. "}
                Recommendations come from a deterministic proxy of a small
                NCCN-lite ruleset (<code>nccn-lite-v0</code>), not TxGemma.
                Treat every recommendation as a discussion prompt, not a
                clinical directive.
              </div>
              <div>
                <span className={`pill ${activeTherapy.provenance.model_state}`}>
                  {activeTherapy.provenance.model_state}
                </span>
                <span className="pill">
                  {activeTherapy.recommended_options.length} options
                </span>
              </div>
              {activeTherapy.recommended_options.slice(0, 5).map((opt, i) => (
                <div key={i} style={{ marginTop: "0.5rem" }}>
                  <strong>{opt.regimen}</strong>{" "}
                  <span style={{ color: "var(--fg-muted)", fontSize: "0.8rem" }}>
                    (line {opt.line_of_therapy})
                  </span>
                  <div style={{ fontSize: "0.85rem" }}>{opt.rationale}</div>
                </div>
              ))}
            </div>
          )}

          {result.elo_ranked_hypotheses.length > 0 && (
            <div className="card">
              <h2>
                L5 Co-Scientist Elo tournament ({result.elo_ranked_hypotheses.length} hypotheses)
              </h2>
              {confirmedTherapy && (
                <div
                  data-testid="elo-stale-notice"
                  style={{
                    background: "#fef3c7",
                    borderLeft: "4px solid #f59e0b",
                    color: "#78350f",
                    padding: "0.5rem 0.75rem",
                    borderRadius: "4px",
                    fontSize: "0.85rem",
                    marginBottom: "0.5rem",
                  }}
                >
                  <strong>Note.</strong> This tournament was scored against the
                  parser-driven pass. Your confirmed receptors above changed the
                  therapy branch but this Elo block was not rerun — re-issue the
                  full case to regenerate hypotheses against the confirmed panel.
                </div>
              )}
              <pre style={{ fontSize: "0.75rem", maxHeight: "300px", overflow: "auto" }}>
                {JSON.stringify(result.elo_ranked_hypotheses, null, 2)}
              </pre>
            </div>
          )}
          <EnvelopeCard env={result} />
        </>
      )}
    </>
  );
}
