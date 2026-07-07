import { useState } from "react";
import { analyzeScreening, type ScreeningFinding, type ScreeningResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";
import { BboxOverlay } from "../components/BboxOverlay";
import { bytesToBase64Chunked } from "../lib/b64";

/** A finding is "probe-driven" when its label starts with `cbis_ddsm_logreg_v1:`
 *  or the CBIS-DDSM/MedSigLIP model name prefix. We split findings into
 *  probe vs zero-shot so the operator sees which one drives `overall_score`.
 *  The label prefix is set by `_run_cbis_ddsm_probe_on_bytes` in api/app.py
 *  and is stable across versions. */
function isProbeFinding(f: ScreeningFinding): boolean {
  return /^cbis_ddsm_logreg_v[0-9]+:/i.test(f.label);
}

/** Format a probe label like "cbis_ddsm_logreg_v1:cancer" as a pill-friendly
 *  string so the operator sees the class without the model version noise. */
function probeClassOf(label: string): string {
  const m = label.match(/^cbis_ddsm_logreg_v[0-9]+:(.+)$/i);
  return m ? m[1] : label;
}

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

  const probeFindings = result ? result.findings.filter(isProbeFinding) : [];
  const zeroShotFindings = result ? result.findings.filter((f) => !isProbeFinding(f)) : [];
  const probeDriven = probeFindings.length > 0;

  return (
    <>
      <div className="card">
        <h2>Screening · mammogram intake</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Upload a DICOM mammogram. The server runs the real preprocessing pipeline
          (laterality, view, breast mask) and — depending on env flags — one
          or more of: MedSigLIP-448 (HAI-DEF, remote Modal GPU),
          the CBIS-DDSM logistic-regression probe on top of a MedSigLIP-448
          embedding (v0.3.0), the SigLIP-baseline proxy, or the MONAI
          heuristic detector. Findings with normalized bboxes are drawn over
          the image.
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
              <div style={{ minWidth: 300 }}>
                <div style={{ marginBottom: "0.5rem" }}>
                  <span className="pill">{result.laterality}</span>
                  <span className="pill">{result.view}</span>
                  <span className="pill">mask={result.breast_mask_coverage.toFixed(3)}</span>
                  {result.orientation_flipped && <span className="pill">flipped</span>}
                </div>

                {/* v0.3.0: overall_score display flags whether it came from
                    the trained probe (recommended) or fell back to zero-shot
                    (informational only). The probe wiring in api/app.py:
                    `_run_cbis_ddsm_probe_on_bytes` overrides overall_score
                    when the probe fires, so any probe finding == probe-driven
                    score. */}
                <div style={{
                  marginBottom: "0.75rem",
                  padding: "0.5rem",
                  border: "1px solid var(--border)",
                  borderRadius: 4,
                  background: probeDriven ? "rgba(233,237,76,0.10)" : "transparent",
                }}>
                  <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Overall score</div>
                  <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>
                    {result.overall_score !== null ? result.overall_score.toFixed(4) : "n/a"}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "var(--fg-muted)" }}>
                    {probeDriven
                      ? "Source: CBIS-DDSM logistic-regression probe (supervised, mammography-specific)"
                      : "Source: MedSigLIP zero-shot argmax (off-label — treat as UNCALIBRATED)"}
                  </div>
                </div>

                {probeDriven && (
                  <div style={{ marginBottom: "0.75rem" }}>
                    <strong style={{ fontSize: "0.85rem" }}>
                      CBIS-DDSM probe finding
                    </strong>
                    <table style={{ width: "100%", fontSize: "0.8rem", marginTop: "0.25rem" }}>
                      <thead><tr>
                        <th style={{ textAlign: "left" }}>class</th>
                        <th style={{ textAlign: "right" }}>P(cancer)</th>
                      </tr></thead>
                      <tbody>
                        {probeFindings.map((f, i) => (
                          <tr key={i}>
                            <td>
                              <span className="pill" style={{
                                background: probeClassOf(f.label) === "cancer" ? "#FD9BED" : "#75A025",
                                color: "black",
                              }}>
                                {probeClassOf(f.label)}
                              </span>
                            </td>
                            <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>
                              {f.score.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{ fontSize: "0.7rem", color: "var(--fg-muted)", marginTop: "0.25rem" }}>
                      Probe: <code>cbis_ddsm_logreg_v1</code> · trained on
                      CBIS-DDSM DDSM mammograms · MedSigLIP-448 embedding
                      dim=1152 · thresholds: default 0.5, Youden 0.5549,
                      recall≥0.85 0.2836
                    </div>
                  </div>
                )}

                <div>
                  <strong style={{ fontSize: "0.85rem" }}>
                    {probeDriven
                      ? `MedSigLIP zero-shot findings (informational, ${zeroShotFindings.length})`
                      : `All findings (${zeroShotFindings.length})`}
                  </strong>
                  <table style={{ width: "100%", fontSize: "0.8rem", marginTop: "0.25rem" }}>
                    <thead><tr>
                      <th style={{ textAlign: "left" }}>label</th>
                      <th style={{ textAlign: "right" }}>score</th>
                      <th>bbox</th>
                    </tr></thead>
                    <tbody>
                      {zeroShotFindings.map((f, i) => (
                        <tr key={i}>
                          <td style={{ fontSize: "0.75rem" }}>{f.label}</td>
                          <td style={{ textAlign: "right", fontFamily: "Menlo, monospace" }}>
                            {f.score.toExponential(3)}
                          </td>
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
