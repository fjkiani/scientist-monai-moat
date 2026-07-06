import { useState } from "react";
import { analyzeScreening, type ScreeningResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";
import { BboxOverlay } from "../components/BboxOverlay";
import { bytesToBase64Chunked } from "../lib/b64";

export function ScreeningTab() {
  const [file, setFile] = useState<File | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ScreeningResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    if (f) {
      // Preview only makes sense for PNG/JPEG (DICOM won't render). Still show it.
      const url = URL.createObjectURL(f);
      setImageDataUrl(url);
    } else {
      setImageDataUrl(null);
    }
  }

  async function submit() {
    if (!file) return;
    setBusy(true); setErr(null); setResult(null);
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const b64 = bytesToBase64Chunked(bytes);
      const r = await analyzeScreening({ dicom_bytes_b64: b64 });
      setResult(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="card">
        <h2>Screening · mammogram intake</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Upload a DICOM mammogram. Server runs the real preprocessing pipeline
          (laterality, view, breast mask) and — depending on env flags — one
          or more of: MedSigLIP-448 (HAI-DEF), SigLIP proxy, MONAI heuristic
          detector. Findings with normalized bboxes are drawn over the image.
        </p>
        <input type="file" accept=".dcm,.dicom,application/dicom,image/*" onChange={onFile} />
        <div style={{ marginTop: "0.75rem" }}>
          <button className="primary" onClick={submit} disabled={!file || busy}>
            {busy ? "Analyzing…" : "Analyze"}
          </button>
        </div>
        {err && <div className="warning" style={{ marginTop: "0.75rem" }}>{err}</div>}
      </div>

      {result && (
        <>
          <div className="card">
            <h2>Image + bboxes</h2>
            <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
              <BboxOverlay imageDataUrl={imageDataUrl} findings={result.findings} />
              <div style={{ minWidth: 260 }}>
                <div style={{ marginBottom: "0.5rem" }}>
                  <span className="pill">{result.laterality}</span>
                  <span className="pill">{result.view}</span>
                  <span className="pill">mask={result.breast_mask_coverage.toFixed(3)}</span>
                </div>
                <div style={{ marginBottom: "0.5rem" }}>
                  <strong>Overall score:</strong>{" "}
                  {result.overall_score !== null ? result.overall_score.toExponential(4) : "n/a"}
                </div>
                <div>
                  <strong>Findings ({result.findings.length})</strong>
                  <table style={{ width: "100%", fontSize: "0.8rem", marginTop: "0.25rem" }}>
                    <thead><tr>
                      <th style={{ textAlign: "left" }}>label</th>
                      <th style={{ textAlign: "right" }}>score</th>
                      <th>bbox</th>
                    </tr></thead>
                    <tbody>
                      {result.findings.map((f, i) => (
                        <tr key={i}>
                          <td>{f.label}</td>
                          <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>{f.score.toExponential(3)}</td>
                          <td>{f.location_bbox_normalized ? "✓" : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          <EnvelopeCard env={result} arbiter={result.arbiter_score} />
        </>
      )}
    </>
  );
}
