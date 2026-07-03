import { useState } from "react";
import { analyzeBiopsy, type BiopsyResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";

export function BiopsyTab() {
  const [file, setFile] = useState<File | null>(null);
  const [reportText, setReportText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<BiopsyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setBusy(true); setErr(null); setResult(null);
    try {
      const payload: any = {};
      if (file) {
        const bytes = new Uint8Array(await file.arrayBuffer());
        payload.wsi_bytes_b64 = btoa(String.fromCharCode(...bytes));
      }
      if (reportText.trim()) payload.report_text = reportText.trim();
      const r = await analyzeBiopsy(payload);
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  return (
    <>
      <div className="card">
        <h2>Biopsy · WSI image or pathology report</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Upload a whole-slide image (WSI patch, gross photo, or JPEG/PNG) and
          the L4b MedSigLIP-448 probe returns a subtype prediction (IDC / DCIS /
          benign) with a calibrated probability. Weights are <strong>synthetic</strong>
          — the honesty warning surfaces this on every response.
        </p>
        <label>WSI (image bytes — no OpenSlide parser)</label>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <label style={{ marginTop: "0.75rem" }}>Pathology report (optional free text)</label>
        <textarea value={reportText} onChange={(e) => setReportText(e.target.value)}
                  placeholder="Free-text pathology report…" />
        <button className="primary" onClick={submit} disabled={busy || (!file && !reportText.trim())}>
          {busy ? "Analyzing…" : "Analyze biopsy"}
        </button>
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
                <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Grade</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>
                  {result.grade ?? "—"}
                </div>
              </div>
            </div>
            <div style={{ marginTop: "0.75rem", fontSize: "0.85rem" }}>
              <strong>Receptor panel</strong> · ER {String(result.receptor_panel.er_positive)} · PR{" "}
              {String(result.receptor_panel.pr_positive)} · HER2 {String(result.receptor_panel.her2_status)}
            </div>
          </div>
          <EnvelopeCard env={result} arbiter={result.arbiter_score} />
        </>
      )}
    </>
  );
}
