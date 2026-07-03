import { useState } from "react";
import { runCaseFull, type FullCaseResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";
import { BboxOverlay } from "../components/BboxOverlay";

/**
 * Runs the end-to-end tumor-board loop by chaining screening → biopsy →
 * therapy on the server. The Elo-ranked hypotheses block is surfaced as
 * raw JSON — the L5 Co-Scientist loop lives on worker-0 and populates it
 * with a real tournament in a later step.
 */
export function CaseViewTab() {
  const [dicomFile, setDicomFile] = useState<File | null>(null);
  const [wsiFile, setWsiFile] = useState<File | null>(null);
  const [reportText, setReportText] = useState("");
  const [age, setAge] = useState<number>(58);
  const [menopausal, setMenopausal] = useState<"pre" | "post" | "peri" | "unknown">("post");
  const [stage, setStage] = useState<string>("T1N0M0");

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<FullCaseResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  async function submit() {
    setBusy(true); setErr(null); setResult(null);
    try {
      const body: any = {
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
        body.biopsy_input = {};
        if (wsiFile) {
          const bytes = new Uint8Array(await wsiFile.arrayBuffer());
          body.biopsy_input.wsi_bytes_b64 = btoa(String.fromCharCode(...bytes));
        }
        if (reportText.trim()) body.biopsy_input.report_text = reportText.trim();
      }
      const r = await runCaseFull(body);
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  return (
    <>
      <div className="card">
        <h2>Case view · end-to-end (screening → biopsy → therapy)</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Chains all three stage endpoints server-side and returns the L5
          Elo-ranked hypotheses block. Any stage can be omitted — the server
          will return that section as <code>null</code>.
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
                      placeholder="Free-text pathology report…" />
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

        <button className="primary" onClick={submit} disabled={busy} style={{ marginTop: "0.75rem" }}>
          {busy ? "Running case…" : "Run full case"}
        </button>
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
            <div className="card">
              <h2>Biopsy summary</h2>
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
            </div>
          )}
          {result.therapy && (
            <div className="card">
              <h2>Therapy summary</h2>
              <div>
                <span className={`pill ${result.therapy.provenance.model_state}`}>
                  {result.therapy.provenance.model_state}
                </span>
                <span className="pill">{result.therapy.recommended_options.length} options</span>
              </div>
              {result.therapy.recommended_options.slice(0, 3).map((opt, i) => (
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
              <h2>L5 Co-Scientist Elo tournament ({result.elo_ranked_hypotheses.length} hypotheses)</h2>
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
