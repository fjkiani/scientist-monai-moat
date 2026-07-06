import { useState } from "react";
import {
  getDemoCase,
  reasonTherapy,
  runCaseFull,
  type FullCaseResponse,
  type TherapyResponse,
} from "../api";
import { base64ToBytes, bytesToBase64Chunked } from "../lib/b64";
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
  const [demoBusy, setDemoBusy] = useState(false);
  const [result, setResult] = useState<FullCaseResponse | null>(null);
  const [confirmedTherapy, setConfirmedTherapy] = useState<TherapyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [demoWarnings, setDemoWarnings] = useState<string[]>([]);

  // v0.2.2: pull a fully-formed demo case from GET /v1/demo/case and
  // populate the form so a first-time user can click Run without hunting
  // for a DICOM. base64 → Uint8Array → File plugs straight into the
  // existing dicomFile state used by submit().
  async function loadDemoCase() {
    setDemoBusy(true);
    setErr(null);
    try {
      const c = await getDemoCase();
      const bytes = base64ToBytes(c.dicom_bytes_b64);
      const file = new File([bytes], "demo.dcm", { type: "application/dicom" });
      setDicomFile(file);
      setImagePreview(URL.createObjectURL(file));
      setReportText(c.report_text);
      if (c.patient_context?.age != null) setAge(Number(c.patient_context.age));
      if (c.patient_context?.menopausal_status) {
        setMenopausal(c.patient_context.menopausal_status);
      }
      if (c.patient_context?.stage_ct) setStage(c.patient_context.stage_ct);
      setDemoWarnings(c.warnings ?? []);
    } catch (e: any) {
      setErr(String(e?.message ?? e ?? "load demo case failed"));
    } finally {
      setDemoBusy(false);
    }
  }

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
          dicom_bytes_b64: bytesToBase64Chunked(bytes),
        };
        setImagePreview(URL.createObjectURL(dicomFile));
      }
      if (wsiFile || reportText.trim()) {
        const bi: Record<string, unknown> = {};
        if (wsiFile) {
          const bytes = new Uint8Array(await wsiFile.arrayBuffer());
          bi.wsi_bytes_b64 = bytesToBase64Chunked(bytes);
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

  // v0.2.2: workflow ribbon — visualizes which pipeline stage each output
  // section corresponds to. State machine:
  //   inputs = the DICOM/report has been provided
  //   run = the /v1/case/full call is in flight
  //   done = a sub-result exists for that stage
  const hasInputs = dicomFile != null || wsiFile != null || reportText.trim().length > 0;
  const stageDone = {
    inputs: hasInputs,
    screening: result?.screening != null,
    biopsy: result?.biopsy != null,
    therapy: activeTherapy != null,
    arbiter: (result?.elo_ranked_hypotheses?.length ?? 0) > 0,
  };

  return (
    <>
      <div className="card" data-testid="workflow-ribbon" style={{ padding: "0.6rem 0.9rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.4rem" }}>
          {[
            { key: "inputs", label: "1. Inputs", hint: "DICOM + report" },
            { key: "screening", label: "2. Screening", hint: "MONAI/MedSigLIP" },
            { key: "biopsy", label: "3. Biopsy + Parse", hint: "regex + probe" },
            { key: "therapy", label: "4. Therapy", hint: "rules-lite / TxGemma" },
            { key: "arbiter", label: "5. Arbiter", hint: "Co-Scientist Elo" },
          ].map((s, i, arr) => {
            const done = stageDone[s.key as keyof typeof stageDone];
            const active = !done && (i === 0 || stageDone[arr[i - 1].key as keyof typeof stageDone]);
            const color = done ? "var(--accent-2, #15803d)" : active ? "var(--accent, #0279EE)" : "var(--fg-muted, #999)";
            return (
              <div key={s.key} style={{ display: "flex", alignItems: "center", flex: "1 1 auto", minWidth: 0 }}>
                <div style={{ display: "flex", flexDirection: "column", flex: "1 1 auto", minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                    <span aria-hidden="true" style={{
                      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
                      background: color, flex: "0 0 auto",
                    }} />
                    <span style={{
                      fontSize: "0.78rem", fontWeight: done ? 700 : 600, color,
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                    }}>{s.label}</span>
                  </div>
                  <span style={{
                    fontSize: "0.68rem", color: "var(--fg-muted, #999)",
                    fontFamily: "Menlo, monospace", marginLeft: 14,
                  }}>{s.hint}</span>
                </div>
                {i < arr.length - 1 && (
                  <span aria-hidden="true" style={{
                    flex: "0 0 auto", width: "0.75rem", textAlign: "center",
                    color: "var(--fg-muted, #ccc)",
                  }}>›</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
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
            onClick={loadDemoCase}
            disabled={demoBusy}
            style={{ background: "var(--accent)", color: "white", border: "1px solid var(--accent)",
                     padding: "0.5rem 0.75rem", cursor: demoBusy ? "wait" : "pointer",
                     opacity: demoBusy ? 0.6 : 1 }}
            data-testid="load-demo-case-caseview"
            title="Fetch a public CBIS-DDSM mammogram + synthetic pathology report from the server"
          >
            {demoBusy ? "Loading demo case…" : "Load demo case (DICOM + report)"}
          </button>
          <button
            type="button"
            onClick={() => setReportText(LUMINAL_A_EXAMPLE)}
            style={{ background: "var(--panel)", border: "1px solid var(--border)",
                     padding: "0.5rem 0.75rem", cursor: "pointer" }}
            data-testid="load-luminal-a-example-caseview"
            title="Populate only the pathology report field with the canned luminal-A text"
          >
            Load ER+ report only (no DICOM)
          </button>
        </div>
        {demoWarnings.length > 0 && (
          <div className="warning" style={{ marginTop: "0.75rem" }} data-testid="demo-warnings">
            <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Demo case loaded — honesty caveats:</div>
            <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
              {demoWarnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          </div>
        )}
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
