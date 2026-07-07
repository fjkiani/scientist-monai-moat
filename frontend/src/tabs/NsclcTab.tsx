import { useState } from "react";
import { runCaseFull, type FullCaseResponse } from "../api";
import { EnvelopeCard } from "../components/Envelope";

/**
 * NSCLC panel.
 *
 * `POST /v1/case/full?cancer=nsclc` returns a `FullCaseResponse` whose
 * `nsclc` block is either a placeholder envelope OR the real LIDC pipeline
 * output (heuristic candidates + optional LUNA16 RetinaNet detections +
 * NCCN-lite therapy). Which branch fires depends on the server env:
 *
 *  - `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1` + a valid `series_dir`
 *    → real pipeline (heuristic OR RetinaNet block)
 *  - `ONCOLOGY_ARBITER_ENABLE_LUNA16_RETINANET=1`
 *    → RetinaNet runs after the heuristic, adds `nsclc.luna16` block
 *
 * We ALWAYS render whichever block is present, but we also render the
 * placeholder path (nsclc=null) so the tab is not blank on a fresh env.
 */
export function NsclcTab() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<FullCaseResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    setBusy(true); setErr(null); setResult(null);
    try {
      // Empty body — the placeholder branch doesn't consume screening/biopsy inputs yet.
      const r = await runCaseFull("nsclc", {});
      setResult(r);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  const nsclc = result?.nsclc ?? null;
  const luna16 = nsclc?.luna16 ?? null;

  return (
    <>
      <div className="card">
        <h2>NSCLC · LIDC-IDRI proxy or LUNA16 RetinaNet</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--fg-muted)" }}>
          Sends <code>POST /v1/case/full?cancer=nsclc</code>. When the server
          is started with <code>ONCOLOGY_ARBITER_ENABLE_LUNA16_RETINANET=1</code>
          AND a real CT series is passed via <code>nsclc_ct_input.series_dir</code>,
          the MONAI RetinaNet <code>lung_nodule_ct_detection@0.6.9</code> runs
          on the volume and its detections appear below. Otherwise the
          response is a placeholder or the HU-threshold heuristic proxy.
        </p>

        <button className="primary" onClick={submit} disabled={busy}
                style={{ marginTop: "0.5rem" }}>
          {busy ? "Running…" : "Run NSCLC pipeline"}
        </button>
        {err && <div className="warning" style={{ marginTop: "0.75rem" }}>{err}</div>}
      </div>

      {result && (
        <>
          <div className="card">
            <h2>NSCLC response summary</h2>
            <div>
              <span className={`pill ${result.provenance.model_state}`}>
                {result.provenance.model_state}
              </span>
              <span className="pill">
                model: {result.provenance.model_name ?? "—"}
              </span>
            </div>
            <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
              screening: <code>{result.screening ? "…" : "null"}</code>
              {" · "}
              biopsy: <code>{result.biopsy ? "…" : "null"}</code>
              {" · "}
              therapy: <code>{result.therapy ? "…" : "null"}</code>
              {" · "}
              elo: <code>{result.elo_ranked_hypotheses.length}</code>
            </div>
          </div>

          {nsclc && (
            <div className="card">
              <h2>NSCLC pipeline result</h2>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <span className={`pill ${nsclc.model_state}`}>{nsclc.model_state}</span>
                <span className="pill">{nsclc.model_name}</span>
              </div>
              <div style={{ marginTop: "0.5rem", fontSize: "0.85rem",
                            color: "var(--fg-muted)" }}>
                lung voxel fraction: <code>{nsclc.lung_voxel_fraction?.toFixed(3) ?? "—"}</code>
                {" · "}
                candidates kept: <code>{nsclc.n_candidates_kept ?? "—"} / {nsclc.n_candidates_total ?? "—"}</code>
                {" · "}
                max diameter: <code>{nsclc.max_diameter_mm?.toFixed(1) ?? "—"} mm</code>
              </div>
              {nsclc.risk_bucket && (
                <div style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
                  Arbiter risk: <strong>{nsclc.risk_bucket}</strong>
                  {" · driving: "}
                  <code>{nsclc.driving_feature ?? "—"}</code>
                  {" · logit "}
                  <code>{nsclc.logit?.toFixed(3) ?? "—"}</code>
                </div>
              )}
              {nsclc.warnings.length > 0 && (
                <ul style={{ marginTop: "0.5rem", fontSize: "0.8rem",
                             color: "var(--fg-muted)", paddingLeft: "1.25rem" }}>
                  {nsclc.warnings.map((w, i) => (
                    <li key={i}><code>{w}</code></li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {luna16 && (
            <div className="card" style={{ borderLeft: "3px solid #75A025" }}>
              <h2>LUNA16 RetinaNet detections</h2>
              <div style={{ background: "#f0fdf4", border: "1px solid #86efac",
                            padding: "0.5rem 0.75rem", marginBottom: "0.75rem",
                            fontSize: "0.8rem", borderRadius: 6 }}>
                <strong>Real trained detector.</strong> MONAI
                {" "}<code>lung_nodule_ct_detection</code> bundle
                {" "}<code>@{luna16.bundle_version}</code>, trained on LUNA16
                (LIDC-IDRI-derived, retrospective, not lung-cancer-labeled).
                RUO — do not read as a cancer probability.
              </div>
              <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                <div>
                  <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Detections</div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>
                    {luna16.n_detections}
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Top score</div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>
                    {luna16.top_score.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div style={{ color: "var(--fg-muted)", fontSize: "0.75rem" }}>Inference</div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 600 }}>
                    {luna16.inference_seconds.toFixed(1)} s
                  </div>
                </div>
              </div>
              {luna16.detections.length > 0 && (
                <table style={{ width: "100%", marginTop: "0.75rem",
                                fontSize: "0.85rem", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", padding: "0.25rem" }}>#</th>
                      <th style={{ textAlign: "left", padding: "0.25rem" }}>Center (z,y,x) mm</th>
                      <th style={{ textAlign: "left", padding: "0.25rem" }}>Diameter mm</th>
                      <th style={{ textAlign: "left", padding: "0.25rem" }}>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {luna16.detections.slice(0, 10).map((d, i) => (
                      <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                        <td style={{ padding: "0.25rem" }}>{i + 1}</td>
                        <td style={{ padding: "0.25rem" }}>
                          <code>
                            {d.center_z_mm.toFixed(1)}, {d.center_y_mm.toFixed(1)}, {d.center_x_mm.toFixed(1)}
                          </code>
                        </td>
                        <td style={{ padding: "0.25rem" }}>
                          <code>{d.diameter_mm.toFixed(1)}</code>
                        </td>
                        <td style={{ padding: "0.25rem" }}>
                          <code>{d.score.toFixed(3)}</code>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {luna16.detections.length > 10 && (
                <div style={{ marginTop: "0.5rem", fontSize: "0.75rem",
                              color: "var(--fg-muted)" }}>
                  Showing 10 of {luna16.detections.length} boxes. Full list is
                  in the JSON payload.
                </div>
              )}
            </div>
          )}

          <EnvelopeCard env={result} />
        </>
      )}
    </>
  );
}
